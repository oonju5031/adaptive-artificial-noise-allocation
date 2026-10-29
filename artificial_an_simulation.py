import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.special import gamma as gamma_fn
from scipy.special import ndtr

# ====================================
# 적응형 AN 전력 할당 시뮬레이션
#
# - 시스템
#   - MISO Wiretap (Alice Nt=2, Bob 1, Eve 1) -> TODO: 이후 N_t > 2인 경우로 확장 예정
#   - 4장: Alice가 Eve의 순시 CSI를 아는 경우 (Genie-aided, 성능 상한)
#   - 5장: Eve의 위치가 불확실한 경우. Eve는 Alice 중심 원 안 어딘가에 면적 균일하게 위치 (원 모델)
# - 비교 (6장)
#   - Genie: Eve의 순시 CSI(위치 포함)를 알고 블록마다 4장 알고리즘으로 phi 선택 -> 상한 (R_s = Cs, 유효 보안 처리량 = Cs의 평균)
#   - Proposed: 위치·g 평균 SOP p_so <= eps 아래 블록마다 phi, R_s 선택
#   - Agnostic-phi: phi는 위치 정보 없이(Eve 수신 잡음 0, 가장 가까운 Eve 가정) 정하고 R_s는 위치 분포로 선택 (p_so = eps)
#                -> Proposed와 phi 선택만 다름 (위치 정보에 맞춘 phi 선택 자체의 가치)
#   - Agnostic: phi, R_s 모두 위치 정보 없이 선택 (기존 연구의 Eve 가정) -> 원 안 모든 위치에서 p_so <= eps
#                -> Agnostic-phi와의 차이는 R_s(보장 수준)의 몫
#   - Static (표로만 출력): Proposed의 phi를 h와 무관하게 하나로 고정, R_s는 블록마다 (p_so = eps)
#                -> Proposed와 phi의 블록 적응 여부만 다름
# - 평가 지표 (논문 표기)
#   - 유효 보안 처리량 E[R_s * 1{누설 없음}] (bps/Hz, Yan 외 2015): 블록마다 R_b = Cb(phi), R_s = [f_eps(phi*)]^+,
#     f_eps(phi) = Cb(phi) - log2(1 + theta_eps(phi)), theta_eps: p_so = eps가 되는 SINR_e 문턱값
#   - 전송 블록의 실제 누설 비율 (MC, 설계한 p_so 확인용)
# ====================================


# ----- HyperParameters --------------
NUM_SAMPLES = 50000                     # Monte-Carlo 채널 표본 수
SEED = 0                                # 난수 시드 (재현성)
SNR_DB_RANGE = np.arange(-20, 31, 5)    # 4장 검증의 송신 SNR 구간 (dB)
RUN_VERIFY = False                      # True면 시뮬레이션 전에 검증 함수 실행 (논문용 검증 수치 출력)

NT = 2              # 송신 안테나 수 (현 공식은 2로 고정된 경우에 한정됨, TODO: 이후 N_t > 2인 경우로 확장 예정)

STATIC_TRAIN_N = 200000   # Static의 phi 계산용 독립 표본 수 (시뮬레이션 표본과 별도)
STATIC_SEED = 12345       # Static 표본 시드 (시뮬레이션 시드와 분리)
PHI_MAX = 0.99      # phi 탐색 상한 (1.0인 경우 신호 전력이 0이므로 제외)

# 5장 (Eve 위치 불확실성): Eve는 Alice 중심 원 안에 면적 균일, 평균 SNR은 거리^-alpha에 비례
PL_ALPHA = 3.0          # 경로 손실 지수
EPS_LIST = (0.01, 0.1)  # SOP 제약 epsilon 대표값
CH5_PHI_FINE_N = 4001   # theta_eps(phi) 사전 계산 격자 점 수 (0~PHI_MAX)
CH5_COARSE_STEP = 20    # 블록별 탐색의 거친 격자 간격 (정밀 격자 index 기준 -> 거친 격자 201점)

# 6장 시뮬레이션 구간
RHO_B_DB_LIST = (10, 20)                    # Bob 송신 SNR rho_b (dB)
RHO_EDGE_DB_RANGE = np.arange(-20, 21, 5)   # 원 가장자리(가장 약한 위치) Eve의 평균 SNR rho_edge (dB), 논문의 rho_e
# ------------------------------------


def hdot(a, b):
    # 표본별 켤레전치 내적 a^H b
    # 이 때 a, b는 [N, Nt] (각 행이 Nt x 1 열벡터 하나)
    return np.sum(np.conj(a) * b, axis=1)


def _check_beamformers(h, w_s, w_z, tol=1e-10):
    # 모델 가정 검증
    # 1. w_s, w_z 정규직교 여부
    # 2. h^H w_z = 0 (영공간)
    assert np.max(np.abs(np.linalg.norm(w_s, axis=1) - 1)) < tol
    assert np.max(np.abs(np.linalg.norm(w_z, axis=1) - 1)) < tol
    assert np.max(np.abs(hdot(w_s, w_z))) < tol, "w_s와 w_z가 직교하지 않음"
    assert np.max(np.abs(hdot(h, w_z))) < tol, "w_z가 h의 영공간에 있지 않음"


def f_ch4(phi, A, B, D):
    # 식 (3.14): f(phi) = Cb(phi) - Ce(phi)  [bps/Hz]
    return (np.log1p(A * (1 - phi)) - np.log1p(B * (1 - phi) / (1 + D * phi))) / np.log(2.0)


def df_ch4(phi, A, B, D):
    # f'(phi)
    return (-A / (1 + A * (1 - phi))
            - ((D - B) / (1 + B + (D - B) * phi) - D / (1 + D * phi))) / np.log(2.0)


def df_numerator_ch4(phi, A, B, D):
    # f'(phi)의 분자 (이차식): f'(phi) = N(phi) / (ln2 * P Q R),
    #   P = 1 + A(1-phi), Q = 1 + B + (D-B)phi, R = 1 + D phi  (구간 [0, 1)에서 모두 양수 -> f'과 N의 부호가 같음)
    #   N(phi) = -A D (D-B) phi^2 - 2 A D (1+B) phi + B(1 + D + A D) - A
    return -A * D * (D - B) * phi**2 - 2 * A * D * (1 + B) * phi + B * (1 + D + A * D) - A


def optimize_genie_closed(A, B, D):
    # 4장: Eve의 순시 CSI(B, D)를 아는 경우 f(phi)를 채널별로 최대화 (닫힌 형태)
    #   f'(phi) = 0의 근은 이차식 N(phi) = 0의 근 (최대 2개) -> 구간 (0, PHI_MAX) 안의 실근과 양 끝점 0, PHI_MAX 중 f가 최대인 점
    #   max f <= 0이면 보안 전송이 불가능하므로 phi* = 0
    a = -A * D * (D - B)
    b = -2 * A * D * (1 + B)
    c = B * (1 + D + A * D) - A
    cand = np.full((len(A), 4), np.nan)
    cand[:, 0] = 0.0
    cand[:, 1] = PHI_MAX
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b**2 - 4 * a * c
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        qq = -0.5 * (b + np.copysign(sq, b))                       # 수치적으로 안정한 근의 공식
        r1 = np.where(a != 0, qq / a, np.nan)
        r2 = np.where(qq != 0, c / qq, np.nan)                      # a = 0 (D = B)이면 일차식 b phi + c = 0의 근
    for k, r in ((2, r1), (3, r2)):
        cand[:, k] = np.where((r > 0) & (r < PHI_MAX), r, np.nan)
    F = np.where(np.isnan(cand), -np.inf, f_ch4(np.nan_to_num(cand), A[:, None], B[:, None], D[:, None]))
    phi = cand[np.arange(len(A)), np.argmax(F, axis=1)]
    return np.where(F.max(axis=1) <= 0, 0.0, phi)


def generate_gains(rng, n, rho_b, rho_e):
    # h, g ~ CN(0, I) 독립 Rayleigh 채널 생성 후 식 (3.8)의 A, B, D 반환 (Nt=2)
    h = (rng.standard_normal((n, NT)) + 1j * rng.standard_normal((n, NT))) / np.sqrt(2)
    g = (rng.standard_normal((n, NT)) + 1j * rng.standard_normal((n, NT))) / np.sqrt(2)
    norm_h = np.linalg.norm(h, axis=1, keepdims=True)
    w_s = h / norm_h                                                            # MRT, 식 (3.2)
    w_z = np.stack([-np.conj(h[:, 1]), np.conj(h[:, 0])], axis=1) / norm_h     # 식 (3.3)
    _check_beamformers(h, w_s, w_z)
    A = rho_b * np.abs(hdot(h, w_s)) ** 2
    B = rho_e * np.abs(hdot(g, w_s)) ** 2
    D = rho_e * np.abs(hdot(g, w_z)) ** 2
    return A, B, D


def verify_ch4_closed(snr_db_list=tuple(SNR_DB_RANGE), num_samples=20000, ref_points=20001, seed=1):
    # 4장 닫힌 형태 검증
    #   1. f'(phi)과 이차식 N(phi)의 부호 일치 (무작위 phi)
    #   2. 닫힌 형태의 해 vs [0, PHI_MAX] 정밀 그리드(ref_points) 탐색: 손실(기준값 - 닫힌 형태), 닫힌 형태가 더 큰 경우의 최대 차이
    #   3. 상한 절단 영향: 탐색 구간을 [0, PHI_MAX]로 제한해 [0, 1)의 최적해를 놓치는 채널 비율과 손실
    rng = np.random.default_rng(seed)
    rng_phi = np.random.default_rng(seed + 1)                      # 부호 확인용 phi 표본 (채널 표본 난수열과 분리)
    ref = np.linspace(0.0, PHI_MAX, ref_points)
    ext = np.linspace(PHI_MAX, 1.0, int(round((1.0 - PHI_MAX) / (ref[1] - ref[0]))) + 1)[1:]
    ref_full = np.concatenate([ref, ext])                          # [0, PHI_MAX] 기준 그리드 + (PHI_MAX, 1] 같은 간격 확장
    print(f"[4장 닫힌 형태 검증] 표본 {num_samples}, 기준 그리드 {ref_points}점 (상한 절단 확인용 확장 {len(ext)}점)")
    print(f"{'SNR(dB)':>7} | {'부호 불일치':>9} | {'손실 최대':>10} | {'초과 최대':>10} | "
          f"{'절단 채널(%)':>11} | {'절단 최대손실':>12} | {'절단 평균손실':>12}")
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        A, B, D = generate_gains(rng, num_samples, rho, rho)
        pr = rng_phi.uniform(0.0, PHI_MAX, num_samples)
        g = df_ch4(pr, A, B, D)
        sign_bad = np.sum((np.sign(g) != np.sign(df_numerator_ch4(pr, A, B, D))) & (np.abs(g) > 1e-12))
        phi = optimize_genie_closed(A, B, D)
        v_c = np.maximum(0, f_ch4(phi, A, B, D))
        v_ref = np.empty(num_samples); v_full = np.empty(num_samples)
        for s0 in range(0, num_samples, 1000):                     # 메모리 절약용 청크
            sl = slice(s0, s0 + 1000)
            F = np.maximum(0, f_ch4(ref_full[None, :], A[sl, None], B[sl, None], D[sl, None]))
            v_full[sl] = F.max(1)
            v_ref[sl] = F[:, :ref_points].max(1)                    # [0, PHI_MAX] 기준값
        diff = v_ref - v_c                                          # > 0: 닫힌 형태가 그리드보다 작음(손실), < 0: 닫힌 형태가 더 큼
        trunc = v_full - v_ref                                      # 상한 절단 손실 (>= 0, 기준값끼리 비교)
        cut = trunc > 0                                             # [0, 1)의 최적해가 (PHI_MAX, 1)에 놓인 채널
        print(f"{snr_db:>7} | {sign_bad:>9} | {max(diff.max(), 0.0):>10.1e} | {max(-diff.min(), 0.0):>10.1e} | "
              f"{cut.mean() * 100:>11.3f} | {trunc.max():>12.1e} | {trunc.mean():>12.1e}")


# ====================================
# 5장: Eve 위치 불확실성 (원 모델)
# - Eve는 Alice 중심 반지름 r0인 원 안 어딘가에 면적 균일하게 위치, 위치는 전송 동안 고정이며 Alice는 모름
#   -> Eve의 평균 SNR = rho_edge * (r/r0)^-alpha (코드의 rho_edge = 논문의 rho_e, 원 가장자리 Eve의 평균 SNR)
# - 블록마다 Alice는 h를 알고 phi, R_b(=Cb), R_s를 정함. 제약: 위치·g 평균 SOP p_so <= eps
# - 성능: 유효 보안 처리량 E[R_s * 1{누설 없음}] (Eve 순시 CSI가 있으면 R_s = Cs, 누설 없음 -> 4장 Genie가 상한)
# - 코드 변수와 논문 표기: tau (L의 인자, tau = theta/(1-phi)), q_* = theta_eps(phi) (SINR_e 문턱값)
# ====================================


def norm_sf(x):
    # 표준정규분포의 꼬리확률 P(Z > x)
    return ndtr(-x)


def disk_L(tau, rho_edge, alpha=PL_ALPHA):
    # 위치 평균 항 L(tau) = E_r[exp(-(tau/rho_e)(r/r0)^alpha)] (거리 r에 대해 평균한 결과)
    # 닫힌형: L(tau) = d (rho_edge/tau)^d * lower_gamma(d, tau/rho_edge), d = 2/alpha (tau -> 0이면 1)
    d = 2.0 / alpha
    x = np.maximum(np.asarray(tau, dtype=float) / rho_edge, 1e-300)
    return d * x ** (-d) * gammainc(d, x) * gamma_fn(d)


def disk_dL(tau, rho_edge, alpha=PL_ALPHA):
    # L'(tau) = -(d/rho_edge) x^(-d-1) lower_gamma(d+1, x), x = tau/rho_edge
    # L'(tau) = (d/tau)(exp(-x) - L(tau))와 같은 식 (본문 식 (5.19)는 여기에 L(tau_eps(0)) = eps를 대입한 형태)
    # 작은 tau에서 두 값의 차가 상쇄되는 것을 피하려고 감마 함수 형태를 사용
    d = 2.0 / alpha
    x = np.maximum(np.asarray(tau, dtype=float) / rho_edge, 1e-300)
    return -(d / rho_edge) * x ** (-d - 1) * gammainc(d + 1, x) * gamma_fn(d + 1)


def eve_ccdf_disk(theta, phi, rho_edge, alpha=PL_ALPHA):
    # 위치·g 평균 꼬리확률 P(SINR_e > theta) = L(tau) / (1 + phi tau), tau = theta/(1-phi)  (Nt=2)
    # 분모(AN의 방해 항)는 위치와 무관, 위치 정보는 L에만 들어감
    tau = theta / (1 - phi)
    return disk_L(tau, rho_edge, alpha) / (1 + phi * tau)


def sop_quantile(phi_grid, eps, L_fun, iters=100):
    # theta_eps(phi): P(SINR_e > theta) = eps가 되는 문턱값, theta_eps = (1-phi) tau
    # tau는 L(tau) = eps (1 + phi tau)의 유일한 근 (좌변 감소, 우변 비감소) -> log tau 이분법 (phi 격자 전체 벡터화)
    # L_fun: 위치 평균 항 L(tau) (rho_e의 분포만 바꾸면 그대로 적용)
    phi = np.asarray(phi_grid, dtype=float)
    lo = np.full(phi.shape, -60.0)
    hi = np.full(phi.shape, 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tau = np.exp(mid)
        left = L_fun(tau) > eps * (1 + phi * tau)                   # 근보다 왼쪽
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    return (1 - phi) * np.exp(0.5 * (lo + hi))


def sop_quantile_disk(phi_grid, eps, rho_edge, alpha=PL_ALPHA):
    # 원 모델의 theta_eps(phi)
    return sop_quantile(phi_grid, eps, lambda tau: disk_L(tau, rho_edge, alpha))


def sop_quantile_fixed(phi_grid, eps, rho_e):
    # Eve의 평균 SNR을 값 하나로 볼 때의 theta_eps(phi) (L(tau) = exp(-tau/rho_e))
    return sop_quantile(phi_grid, eps, lambda tau: np.exp(-tau / rho_e))


def sop_quantile_noiseless(phi_grid, eps):
    # Eve 수신 잡음 0 (가장 가까운 Eve, 평균 SNR -> inf): P(SINR_e > theta) = 1/(1 + theta phi/(1-phi))
    # -> theta_eps = (1-phi)(1-eps)/(phi eps). 원 안 모든 위치에서 p_so <= eps를 보장 (Agnostic 기법)
    phi = np.asarray(phi_grid, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(phi > 0, (1 - phi) * (1 - eps) / (phi * eps), np.inf)   # phi = 0이면 보안 전송 불가


def harmonic_mean_disk(rho_edge, alpha=PL_ALPHA):
    # 원 모델에서 Eve 평균 SNR의 조화평균 1/E[1/rho] = (1 + alpha/2) rho_edge (산술평균은 발산)
    # rho_eq(tau)의 tau -> 0 극한
    return (1 + alpha / 2) * rho_edge


def rho_eq_disk(tau, rho_edge, alpha=PL_ALPHA):
    # 등가 SNR rho_eq(tau) = -tau / ln L(tau): 같은 tau에서 L(tau)와 같은 값을 주는 단일 평균 SNR
    # 분위수 방정식 L(tau) = eps(1 + phi tau)에서 L(tau) = exp(-tau/rho_eq)로 바꿔도 근이 같음
    # -> theta_eps(phi)는 평균 SNR이 rho_eq(tau(phi))인 고정 Eve의 문턱값과 정확히 같음
    # ln L(tau)가 tau에 대해 볼록(누적생성함수) -> rho_eq는 tau에 대해 증가, tau -> 0이면 조화평균, tau -> inf이면 상한 없이 증가
    return -np.asarray(tau, dtype=float) / np.log(disk_L(tau, rho_edge, alpha))


def an_on_threshold_disk(eps, rho_edge, alpha=PL_ALPHA):
    # AN 사용 조건: A/(1+A) < T 이면 phi* > 0 (f_eps(phi)의 phi = 0에서의 기울기 > 0)
    #   T = tau0 (1 + eps/|L'(tau0)|) / (1 + tau0),  tau0 = L^{-1}(eps) (phi = 0이면 theta = tau)
    # T >= 1이면 모든 A에서 AN 사용
    tau0 = sop_quantile_disk(np.array([0.0]), eps, rho_edge, alpha)[0]
    return tau0 * (1 + eps / abs(disk_dL(tau0, rho_edge, alpha))) / (1 + tau0)


def generate_gains_disk(rng, n, rho_b, rho_edge, alpha=PL_ALPHA):
    # 표본마다 Eve 위치를 원 안 면적 균일로 생성: (r/r0)^2 = u ~ U(0, 1] -> 평균 SNR = rho_edge * u^(-alpha/2)
    u = 1.0 - rng.uniform(size=n)                                   # (0, 1] (u = 0 제외)
    rho_e = rho_edge * u ** (-alpha / 2)
    A, B, D = generate_gains(rng, n, rho_b, rho_e)                  # B, D는 표본별 rho_e로 계산 (식 3.8)
    return A, B, D, rho_e


def optimize_ch5_disk(A, phi_fine, q_fine, coarse_step=CH5_COARSE_STEP, chunk=20000):
    # 블록별 f_eps(phi) = log2(1 + A(1-phi)) - log2(1 + theta_eps(phi)) 최대화, R_s = [f_eps(phi*)]^+
    # (q_fine = theta_eps(phi)는 h와 무관 -> 사전 계산해 공유)
    # f_eps의 내부 극대점은 하나 이하(수치 확인) -> 거친 격자 argmax 후 이웃 구간에서 정밀 격자 argmax
    n = len(A)
    idx_c = np.arange(0, len(phi_fine), coarse_step)
    lq = np.log2(1 + q_fine)
    offs = np.arange(-coarse_step, coarse_step + 1)
    phi_opt = np.empty(n)
    rs_opt = np.empty(n)
    for s0 in range(0, n, chunk):
        a = A[s0:s0 + chunk, None]
        rc = np.log2(1 + a * (1 - phi_fine[idx_c][None, :])) - lq[idx_c][None, :]
        kc = idx_c[rc.argmax(1)]
        win = np.clip(kc[:, None] + offs[None, :], 0, len(phi_fine) - 1)
        rf = np.log2(1 + a * (1 - phi_fine[win])) - lq[win]
        j = rf.argmax(1)
        rows = np.arange(len(kc))
        phi_opt[s0:s0 + chunk] = phi_fine[win[rows, j]]
        rs_opt[s0:s0 + chunk] = np.maximum(0.0, rf[rows, j])         # R_s <= 0이면 전송하지 않음
    return phi_opt, rs_opt


def rs_at_phi(A, phi, phi_fine, q_fine):
    # 정해진 phi(블록별 또는 고정)에서 실제 위치 분포의 theta_eps로 정한 R_s = [log2(1 + A(1-phi)) - log2(1 + theta_eps(phi))]^+
    return np.maximum(0.0, np.log2(1 + A * (1 - phi)) - np.log2(1 + np.interp(phi, phi_fine, q_fine)))


def optimize_static_ch5(rho_b, phi_fine, q_fine, coarse_step=CH5_COARSE_STEP, n=STATIC_TRAIN_N, seed=STATIC_SEED, chunk=20000):
    # Static: h 순시값 없이 phi 하나로 E_h[R_s(phi)] 최대화
    # ||h||^2 ~ Gamma(Nt, 1) 독립 표본으로 E_h를 추정 (평가 표본과 분리해 편향 방지), 거친 격자 후 이웃 정밀 격자
    rng = np.random.default_rng(seed)
    A = rho_b * rng.gamma(NT, 1.0, size=n)
    lq = np.log2(1 + q_fine)

    def mean_rs(idx):
        acc = np.zeros(len(idx))
        for s0 in range(0, n, chunk):
            a = A[s0:s0 + chunk, None]
            acc += np.maximum(0.0, np.log2(1 + a * (1 - phi_fine[idx][None, :])) - lq[idx][None, :]).sum(0)
        return acc

    idx_c = np.arange(0, len(phi_fine), coarse_step)
    kc = idx_c[np.argmax(mean_rs(idx_c))]
    idx_f = np.clip(np.arange(kc - coarse_step, kc + coarse_step + 1), 0, len(phi_fine) - 1)
    return phi_fine[idx_f[np.argmax(mean_rs(idx_f))]]


def verify_ch5_disk(rho_b_db_list=RHO_B_DB_LIST, rho_edge_db_list=(-10, 0, 10), eps_list=EPS_LIST,
                    num_samples=200000, seed=3, alpha=PL_ALPHA):
    # 5장 해석식 검증
    #   A. 위치 평균 항 L(tau): 닫힌형 vs 수치적분 (Gauss-Legendre)
    #   A2. 이분법으로 구한 tau_eps의 상대 오차 (정밀 격자 전체, 40자리 정밀도 계산 기준, mpmath가 있을 때만, 약 2.5분)
    #   B. 위치·g 평균 Eve SINR 꼬리확률: 해석식 vs MC (실제 h, g, 위치 표본)
    #   C. 설계한 (phi*, R_b, R_s)에서 전송 블록의 실제 누설 비율(MC)이 eps와 같은지
    #   D. AN 사용 조건 A/(1+A) < T: 닫힌형 판정 vs 정밀 격자 전수 탐색
    #   E. 등가 SNR rho_eq: 분위수 등가성, 단조성, 조화평균 극한
    #   F. f_eps(phi)의 단봉성(정밀 격자에서 내부 극대 <= 1, 내부 극소 = 0)과 블록별 탐색(거친 격자 + 이웃 정밀 격자)의 손실 (A 전 구간)
    #   G. 정밀 격자 간격에 따른 R_s 손실 (4001점 vs 40001점)
    rng = np.random.default_rng(seed)
    glx, glw = np.polynomial.legendre.leggauss(4000)
    u, w = (glx + 1) / 2, glw / 2
    print(f"[5장 해석식 검증 A] 위치 평균 항 L(tau): 닫힌형 vs 수치적분 (alpha = {alpha})")
    print(f"{'rho_edge(dB)':>12} | {'최대 상대오차':>12}")
    for re_db in rho_edge_db_list:
        re = 10 ** (re_db / 10)
        ss = np.geomspace(1e-3, 1e3, 25) * re
        num = np.array([np.sum(w * np.exp(-s * u ** (alpha / 2) / re)) for s in ss])
        print(f"{re_db:>12} | {np.max(np.abs(disk_L(ss, re, alpha) / num - 1)):>12.1e}")

    try:
        import mpmath as mp
    except ImportError:
        mp = None
    if mp is None:
        print("\n[5장 해석식 검증 A2] mpmath가 없어 건너뜀")
    else:
        mp.mp.dps = 40
        print(f"\n[5장 해석식 검증 A2] 이분법으로 구한 tau_eps의 상대 오차 (정밀 격자 {CH5_PHI_FINE_N}점 전체, 40자리 정밀도 기준)")
        print(f"{'eps':>5} | {'최대 상대오차':>12}")
        d_mp = mp.mpf(2) / mp.mpf(alpha)
        phi_grid = np.linspace(0.0, PHI_MAX, CH5_PHI_FINE_N)
        for eps in eps_list:
            worst = 0.0
            for re_db in RHO_EDGE_DB_RANGE:
                re = 10 ** (re_db / 10)
                R = mp.mpf(re)
                taus = sop_quantile_disk(phi_grid, eps, re, alpha) / (1 - phi_grid)
                for phi, t in zip(phi_grid, taus):
                    g = lambda x: d_mp * (R / x) ** d_mp * mp.gammainc(d_mp, 0, x / R) - eps * (1 + phi * x)
                    t_ref = mp.findroot(g, mp.mpf(t))
                    worst = max(worst, abs(float(t / t_ref - 1)))
            print(f"{eps:>5.2f} | {worst:>12.1e}")

    print(f"\n[5장 해석식 검증 B] Eve SINR 꼬리확률: 해석식 vs MC (표본 {num_samples})")
    print(f"{'rho_edge(dB)':>12} | {'phi':>4} | {'P 해석':>9} | {'P MC':>9} | {'z':>5}")
    for re_db in rho_edge_db_list:
        re = 10 ** (re_db / 10)
        _, B, D, _ = generate_gains_disk(rng, num_samples, 1.0, re, alpha)
        for phi in (0.1, 0.5, 0.9):
            for p_target in (0.5, 0.1, 0.01):
                t = sop_quantile_disk(np.array([phi]), p_target, re, alpha)[0]   # 꼬리확률이 p_target이 되는 t
                p_an = eve_ccdf_disk(t, phi, re, alpha)
                p_mc = np.mean(B * (1 - phi) / (1 + D * phi) > t)
                z = abs(p_an - p_mc) / np.sqrt(p_an * (1 - p_an) / num_samples)
                print(f"{re_db:>12} | {phi:>4.1f} | {p_an:>9.5f} | {p_mc:>9.5f} | {z:>5.2f}")

    print(f"\n[5장 해석식 검증 C] 설계한 (phi*, R_b, R_s)에서 전송 블록의 누설 비율 (조합마다 독립 표본 {num_samples})")
    print(f"{'rho_b(dB)':>9} | {'rho_edge(dB)':>12} | {'eps':>5} | {'전송 블록 수':>10} | {'누설 비율 MC':>11} | {'z':>6} | {'유효 처리량':>10}")
    phi_fine = np.linspace(0.0, PHI_MAX, CH5_PHI_FINE_N)
    z_all, neps_all = [], []
    for rb_db in rho_b_db_list:                                         # 3.6절 설정 (rho_b, rho_e 전 구간)
        rho_b = 10 ** (rb_db / 10)
        for re_db in RHO_EDGE_DB_RANGE:
            re = 10 ** (re_db / 10)
            for eps in eps_list:
                A, B, D, _ = generate_gains_disk(rng, num_samples, rho_b, re, alpha)   # 조합마다 새 표본 -> 조합 간 z 독립
                q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
                phi_opt, rs = optimize_ch5_disk(A, phi_fine, q_fine)
                tx = rs > 0
                q_opt = np.interp(phi_opt, phi_fine, q_fine)          # phi_opt는 격자점 -> 정확히 theta_eps(phi*)
                leak_all = tx & (B * (1 - phi_opt) / (1 + D * phi_opt) > q_opt)   # Ce > R_b - R_s  <=>  SINR_e > theta_eps(phi*)
                est = np.mean(rs * ~leak_all)                          # 유효 보안 처리량 (샌 블록은 0)
                n_tx = int(tx.sum())
                if n_tx == 0:                                          # 전송 블록 없음 -> 누설 비율 정의되지 않음
                    print(f"{rb_db:>9} | {re_db:>12} | {eps:>5.2f} | {n_tx:>10} | {'-':>11} | {'-':>6} | {est:>10.4f}")
                    continue
                p_leak = leak_all[tx].mean()
                z = (p_leak - eps) / np.sqrt(eps * (1 - eps) / n_tx)  # 부호 있는 z (본문 정의와 같음)
                z_all.append(z)
                neps_all.append(n_tx * eps)
                print(f"{rb_db:>9} | {re_db:>12} | {eps:>5.2f} | {n_tx:>10} | {p_leak:>11.5f} | {z:>6.2f} | {est:>10.4f}")
    z_max = np.max(np.abs(z_all))
    p_one = 2 * norm_sf(z_max)
    print(f"  |z| 최댓값 {z_max:.2f}, 독립인 {len(z_all)}개 조합에서 표본 오차만으로 이 값 이상이 나올 확률 {1 - (1 - p_one) ** len(z_all):.3f}, "
          f"최소 기대 누설 블록 수(n*eps) {min(neps_all):.1f}")

    A_wide = 10 ** (np.arange(-20, 50.001, 0.05) / 10)                # D, F, G 공통 A 범위: -20 ~ 50 dB, 0.05 dB 간격
    print(f"\n[5장 해석식 검증 D] AN 사용 조건 A/(1+A) < T: 닫힌형 판정 vs 정밀 격자 전수 탐색 (A = -20 ~ 50 dB, 0.05 dB 간격)")
    print(f"{'rho_edge(dB)':>12} | {'eps':>5} | {'T':>9} | {'AN 사용 A 구간':>16} | {'불일치':>9}")
    A_sweep = A_wide
    for re_db in RHO_EDGE_DB_RANGE:
        re = 10 ** (re_db / 10)
        for eps in eps_list:
            T = an_on_threshold_disk(eps, re, alpha)
            q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
            R = np.log2(1 + A_sweep[:, None] * (1 - phi_fine[None, :])) - np.log2(1 + q_fine)[None, :]
            brute_on = R.max(1) > R[:, 0] + 1e-12                              # phi > 0에서 더 큰 값이 있음
            pred_on = A_sweep / (1 + A_sweep) < T
            rng_txt = "모든 A" if T >= 1 else f"A < {10 * np.log10(T / (1 - T)):.2f} dB"
            print(f"{re_db:>12} | {eps:>5.2f} | {T:>9.3f} | {rng_txt:>16} | {int(np.sum(brute_on != pred_on)):>3}/{len(A_sweep)}")

    print(f"\n[5장 해석식 검증 E] 등가 SNR rho_eq: 분위수 등가성, 단조성, 조화평균 극한")
    print(f"{'rho_edge(dB)':>12} | {'eps':>5} | {'theta 등가 최대 상대오차':>18} | {'단조 증가':>8} | {'rho_eq(tau->0)/조화평균':>21}")
    tau_scan = np.geomspace(1e-6, 1e6, 200)
    for re_db in rho_edge_db_list:
        re = 10 ** (re_db / 10)
        r_eq = rho_eq_disk(tau_scan * re, re, alpha)
        mono = bool(np.all(np.diff(r_eq) > 0))
        lim = r_eq[0] / harmonic_mean_disk(re, alpha)
        for eps in eps_list:
            phis = np.linspace(0.0, 0.95, 20)
            q_disk = sop_quantile_disk(phis, eps, re, alpha)
            tau_phi = q_disk / (1 - phis)
            q_equiv = np.array([sop_quantile_fixed(np.array([p]), eps, rho_eq_disk(t, re, alpha))[0] for p, t in zip(phis, tau_phi)])
            print(f"{re_db:>12} | {eps:>5.2f} | {np.max(np.abs(q_equiv / q_disk - 1)):>18.1e} | {str(mono):>8} | {lim:>21.6f}")

    print(f"\n[5장 해석식 검증 F] f_eps(phi)의 단봉성과 블록별 탐색의 손실 (A = -20 ~ 50 dB, 0.05 dB 간격, {len(A_wide)}개)")
    print(f"{'rho_edge(dB)':>12} | {'eps':>5} | {'내부 극대 최대':>12} | {'내부 극소 최대':>12} | {'손실 최대':>10}")
    for re_db in RHO_EDGE_DB_RANGE:
        re = 10 ** (re_db / 10)
        for eps in eps_list:
            q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
            R = np.log2(1 + A_wide[:, None] * (1 - phi_fine[None, :])) - np.log2(1 + q_fine)[None, :]
            n_max = np.zeros(len(A_wide), int)
            n_min = np.zeros(len(A_wide), int)
            for i, s in enumerate(np.sign(np.diff(R, axis=1))):
                s = s[s != 0]
                ch = np.diff(s)
                n_max[i] = np.sum(ch < 0)                                   # 증가 -> 감소: 내부 극대
                n_min[i] = np.sum(ch > 0)                                   # 감소 -> 증가: 내부 극소
            _, rs = optimize_ch5_disk(A_wide, phi_fine, q_fine)
            loss = (np.maximum(0.0, R.max(1)) - rs).max()
            print(f"{re_db:>12} | {eps:>5.2f} | {n_max.max():>12} | {n_min.max():>12} | {loss:>10.1e}")

    ref_n = 10 * (CH5_PHI_FINE_N - 1) + 1                              # 10배 조밀한 기준 격자
    phi_ref = np.linspace(0.0, PHI_MAX, ref_n)
    A_h = A_wide                                                       # F와 같은 A 범위 (-20 ~ 50 dB, 0.05 dB 간격)
    print(f"\n[5장 해석식 검증 G] 정밀 격자 간격에 따른 R_s 손실 ({CH5_PHI_FINE_N}점 vs {ref_n}점, A = -20 ~ 50 dB, 0.05 dB 간격)")
    print(f"{'eps':>5} | {'손실 최대':>10}")
    for eps in eps_list:
        worst = 0.0
        for re_db in RHO_EDGE_DB_RANGE:
            re = 10 ** (re_db / 10)
            _, rs = optimize_ch5_disk(A_h, phi_fine, sop_quantile_disk(phi_fine, eps, re, alpha))
            lq_ref = np.log2(1 + sop_quantile_disk(phi_ref, eps, re, alpha))
            best = np.empty(len(A_h))
            for s0 in range(0, len(A_h), 50):                              # 메모리 절약용 청크
                a = A_h[s0:s0 + 50, None]
                best[s0:s0 + 50] = np.maximum(0.0, (np.log2(1 + a * (1 - phi_ref[None, :])) - lq_ref[None, :]).max(1))
            worst = max(worst, (best - rs).max())
        print(f"{eps:>5.2f} | {worst:>10.1e}")


# ====================================
# 6장: 원 모델에서의 기법 비교
# ====================================
SCHEMES = ("genie", "proposed", "agnostic_phi", "agnostic", "static")
PLOT_SCHEMES = ("genie", "proposed", "agnostic_phi", "agnostic")     # Static은 Proposed와 겹쳐 표로만 출력
SCHEME_STYLE = {   # 라벨, 색, 마커·선 (기법별 통일, 흑백 인쇄에서도 구분되도록 그림에 쓰는 기법끼리 선 모양이 겹치지 않게 지정)
    "genie":         ("Genie", "#10b981", "^-."),
    "proposed":      ("Proposed", "#3b82f6", "s-"),
    "agnostic_phi": ("Agnostic-$\\phi$", "#f59e0b", "o--"),
    "agnostic":     ("Agnostic", "#ef4444", "v:"),
    "static":        ("Static", "#8b5cf6", "x-"),   # 그림에는 쓰지 않음 (Proposed와 겹침)
}
RHO_E_LABEL = "Average SNR of Eve at disk edge, $\\rho_e$ (dB)"


def run_simulation(num_samples=NUM_SAMPLES, rho_b_db_list=RHO_B_DB_LIST, rho_edge_db_range=RHO_EDGE_DB_RANGE,
                   eps_list=EPS_LIST, seed=SEED, alpha=PL_ALPHA):
    # 원 모델에서 기법별 유효 보안 처리량, 전송 블록의 평균 phi, 전송 블록의 실제 누설 비율(MC)
    # 유효 보안 처리량 = E[R_s * 1{누설 없음}] (Yan 외 2015): 샌 블록은 0으로 셈 -> 어떤 eps에서도 Genie(E[Cs])가 상한
    # (rho_b, rho_edge)마다 채널·위치 표본을 한 번 생성해 eps와 모든 기법이 공유
    rng = np.random.default_rng(seed)
    phi_fine = np.linspace(0.0, PHI_MAX, CH5_PHI_FINE_N)
    q_agnostic = {eps: sop_quantile_noiseless(phi_fine, eps) for eps in eps_list}  # rho_edge와 무관
    res = {(rb, eps): {"rho_edge": [], **{f"{m}_{k}": [] for m in ("eta", "raw", "phi", "leak") for k in SCHEMES}}
           for rb in rho_b_db_list for eps in eps_list}

    print(f"시뮬레이션 시작 (Samples: {num_samples}, alpha: {alpha})")
    for rb_db in rho_b_db_list:
        rho_b = 10 ** (rb_db / 10)
        for re_db in rho_edge_db_range:
            re = 10 ** (re_db / 10)
            A, B, D, _ = generate_gains_disk(rng, num_samples, rho_b, re, alpha)

            # Genie: Eve 순시 CSI(위치 포함)로 블록마다 4장 알고리즘 -> R_s = [Cb - Ce]^+ (eps와 무관, 누설 없음)
            phi_g = optimize_genie_closed(A, B, D)
            rs_g = np.maximum(0.0, f_ch4(phi_g, A, B, D))

            for eps in eps_list:
                q_true = sop_quantile_disk(phi_fine, eps, re, alpha)                    # 실제 위치 분포의 분위수
                phi_n, rs_n = optimize_ch5_disk(A, phi_fine, q_agnostic[eps])          # Agnostic: Eve 수신 잡음 0(가장 가까운 Eve) 가정 설계
                phi_s = optimize_static_ch5(rho_b, phi_fine, q_true)
                out = {
                    "genie":         (phi_g, rs_g),
                    "proposed":      optimize_ch5_disk(A, phi_fine, q_true),
                    "agnostic_phi": (phi_n, rs_at_phi(A, phi_n, phi_fine, q_true)),    # Agnostic-phi: phi는 수신 잡음 0 가정, R_s는 실제 분포
                    "agnostic":     (phi_n, rs_n),                                     # Agnostic: phi, R_s 모두 수신 잡음 0 가정
                    "static":        (np.full(num_samples, phi_s), rs_at_phi(A, phi_s, phi_fine, q_true)),
                }
                r = res[(rb_db, eps)]
                r["rho_edge"].append(re_db)
                for k, (phi_k, rs_k) in out.items():
                    tx = rs_k > 0
                    if k == "genie":
                        leak = np.zeros(num_samples, dtype=bool)                       # Eve CSI로 R_s = Cs -> 누설 없음
                    else:
                        thr = (1 + A * (1 - phi_k)) / 2 ** rs_k - 1                    # 2^(R_b - R_s) - 1
                        leak = tx & (B * (1 - phi_k) / (1 + D * phi_k) > thr)          # Ce > R_b - R_s
                    r[f"eta_{k}"].append(np.mean(rs_k * ~leak))                       # 유효 보안 처리량: 샌 블록은 0
                    r[f"raw_{k}"].append(rs_k.mean())                                  # 샌 블록 포함 (Zhang 외의 정의, 비교용)
                    r[f"phi_{k}"].append(phi_k[tx].mean() if tx.any() else np.nan)
                    r[f"leak_{k}"].append(leak[tx].mean() if tx.any() else np.nan)

    # ----- 결과 표 출력 -----
    for (rb_db, eps), r in res.items():
        print(f"\n[rho_b = {rb_db} dB, eps = {eps}] 유효 보안 처리량 (bps/Hz), 전송 블록 평균 phi, 전송 블록의 실제 누설 비율")
        print(f"{'rho_edge':>8} | " + " | ".join(f"{k:>13}" for k in SCHEMES)
              + f" | {'phi(P)':>6} | {'phi(A)':>6} | {'누설(P)':>7} | {'누설(A-phi)':>9} | {'누설(A)':>7}")
        for i, re_db in enumerate(r["rho_edge"]):
            print(f"{re_db:>8} | " + " | ".join(f"{r[f'eta_{k}'][i]:>13.4f}" for k in SCHEMES)
                  + f" | {r['phi_proposed'][i]:>6.3f} | {r['phi_agnostic'][i]:>6.3f} | {r['leak_proposed'][i]:>7.4f}"
                  + f" | {r['leak_agnostic_phi'][i]:>9.4f} | {r['leak_agnostic'][i]:>7.4f}")
        gap = 100 * (1 - np.array(r["eta_static"]) / np.array(r["eta_proposed"]))
        print(f"  Static의 Proposed 대비 손실: 최대 {np.nanmax(gap):.2f}% (phi 블록 적응의 이득)")
        ok = np.array(r["raw_proposed"]) > 1e-3
        ratio = np.array(r["eta_proposed"])[ok] / np.array(r["raw_proposed"])[ok]
        print(f"  Proposed의 (유효 / 샌 블록 포함) 비: {ratio.min():.4f} ~ {ratio.max():.4f} (이론값 1 - eps = {1 - eps:.2f})")
    return res


def _plot_grid(res, key, schemes, ylabel, rho_b_db_list, eps_list, ylim=(0, None)):
    # (rho_b, eps) 조합별 작은 그림 (a), (b), ...: 가로축 rho_e, 범례는 그림 위에 한 줄 (모든 작은 그림에 공통)
    nr, nc = len(rho_b_db_list), len(eps_list)
    fig, axs = plt.subplots(nr, nc, figsize=(6.5 * nc, 4.8 * nr), squeeze=False)
    for i, rb_db in enumerate(rho_b_db_list):
        for j, eps in enumerate(eps_list):
            ax, r = axs[i, j], res[(rb_db, eps)]
            for k in schemes:
                label, color, style = SCHEME_STYLE[k]
                ax.plot(r["rho_edge"], r[f"{key}_{k}"], style, color=color, label=label, linewidth=1.6,
                        markersize=6, markerfacecolor="none", markeredgewidth=1.4)
            ax.set_title(f"({'abcdefgh'[i * nc + j]}) $\\rho_b$ = {rb_db} dB, $\\varepsilon$ = {eps}")
            ax.set_xlabel(RHO_E_LABEL)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)                                              # 세로축 0에서 시작 (비율이 그대로 읽히도록)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.03 * (hi - lo), hi)                          # 0 근처 곡선이 가로축 선에 묻히지 않도록 아래쪽 여백 3%
            ax.grid(True, linestyle=":", alpha=0.7)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(schemes), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))                                  # 위쪽은 범례 자리
    return fig


def plot_results(res, rho_b_db_list=RHO_B_DB_LIST, eps_list=EPS_LIST):
    plt.rcParams["mathtext.fontset"] = "stix"                               # 수식 기호 서체 (rho, phi, epsilon 등)

    # ===== (1) rho_edge에 따른 유효 보안 처리량 (그림 6.1) =====
    _plot_grid(res, "eta", PLOT_SCHEMES, "Effective Secrecy Throughput, $\\overline{R}_s$ (bps/Hz)",
               rho_b_db_list, eps_list)

    # ===== (2) rho_edge에 따른 평균 phi (전송 블록 기준) =====
    _plot_grid(res, "phi", ("genie", "proposed", "agnostic"), "Average $\\phi$ (transmitted blocks)",
               rho_b_db_list, eps_list, ylim=(0, 1))
    plt.show()


if __name__ == "__main__":
    # 검증 (RUN_VERIFY = True일 때만)
    if RUN_VERIFY:
        verify_ch4_closed()
        verify_ch5_disk()

    # 시뮬레이션 수행
    plot_results(run_simulation())
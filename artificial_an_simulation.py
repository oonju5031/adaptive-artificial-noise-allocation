import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.special import gamma as gamma_fn

# ====================================
# 적응형 AN 전력 할당 시뮬레이션
#
# - 시스템
#   - MISO Wiretap (Alice Nt=2, Bob 1, Eve 1) -> TODO: 이후 N_t > 2인 경우로 확장 예정
#   - 4장: Alice가 Eve의 순시 CSI를 아는 경우 (Genie-aided, 성능 상한)
#   - 5장: Eve의 위치가 불확실한 경우. Eve는 Alice 중심 원 안 어딘가에 면적 균일하게 위치 (원 모델)
# - 비교 (6장)
#   - Genie-aided: Eve의 순시 CSI(위치 포함)를 알고 블록마다 4장 알고리즘으로 phi 선택 -> 상한 E[Cs]
#   - Proposed: 위치·g 평균 SOP <= eps 아래 블록마다 phi, R_s 선택
#   - Noiseless: Eve 수신 잡음 0 가정(기존 연구의 Eve 가정)으로 블록마다 phi, R_s 선택
#                -> 원 안 모든 위치에서 SOP <= eps 보장. Proposed와 Eve 가정만 다름 (위치 정보의 가치)
#   - Noiseless-phi: Noiseless의 phi를 쓰고 R_s는 실제 위치 분포로 선택 (SOP = eps)
#                -> Proposed와 phi 선택만 다름 (위치 정보에 맞춘 phi 선택 자체의 가치)
#   - Static: Proposed의 phi를 h와 무관하게 하나로 고정, R_s는 블록마다 (SOP = eps)
#                -> Proposed와 phi의 블록 적응 여부만 다름 (결과는 표로만 출력)
# - 평가 지표
#   - 보안 처리량 E_h[R_s] (bps/Hz): 블록마다 R_b = Cb(phi), R_s = R_b - log2(1 + q_eps(phi)), R_s <= 0이면 전송 안 함
#   - 전송 블록의 실제 누설 비율 (MC, 설계한 SOP 확인용)
# ====================================


# ----- HyperParameters --------------
NUM_SAMPLES = 50000                     # Monte-Carlo 채널 표본 수
NUM_PHI = 201                           # Grid search 해상도(정밀도)
SEED = 0                                # 난수 시드 (재현성)
SNR_DB_RANGE = np.arange(-20, 31, 5)    # 4장 검증의 송신 SNR 구간 (dB)
RUN_VERIFY = False                      # True면 시뮬레이션 전에 검증 함수 실행 (논문용 검증 수치 출력)

NT = 2              # 송신 안테나 수 (현 공식은 2로 고정된 경우에 한정됨, TODO: 이후 N_t > 2인 경우로 확장 예정)

STATIC_TRAIN_N = 200000   # Static의 phi 계산용 독립 표본 수 (시뮬레이션 표본과 별도)
STATIC_SEED = 12345       # Static 표본 시드 (시뮬레이션 시드와 분리)
PHI_MAX = 0.99      # phi 탐색 상한 (1.0인 경우 신호 전력이 0이므로 제외)

# 뉴턴법 (4장, 구간 이탈 또는 수렴 지연 시 이분법으로 보완)
NEWTON_INIT = 0.3       # 뉴턴법 초기값
NEWTON_MAX_ITER = 50    # 최대 반복 수 (이분법 대체 포함)
NEWTON_TOL = 1e-9       # 수렴 판정 (phi 변화량)
NEWTON_GTOL = 1e-12     # 수렴 판정 (|f'| 크기)

# 낮은 해상도 그리드 (4장 불확실성 구간)
LOW_RES_GRID_N = 11     # 낮은 해상도 그리드 점 수 (0~PHI_MAX 균등 분할)

# 5장 (Eve 위치 불확실성): Eve는 Alice 중심 원 안에 면적 균일, 평균 SNR은 거리^-alpha에 비례
PL_ALPHA = 3.0          # 경로 손실 지수
EPS_LIST = (0.01, 0.1)  # SOP 제약 epsilon 대표값
CH5_PHI_FINE_N = 4001   # q_eps(phi) 사전 계산 격자 점 수 (0~PHI_MAX)
CH5_COARSE_STEP = 20    # 블록별 탐색의 거친 격자 간격 (정밀 격자 index 기준 -> 거친 격자 201점)

# 6장 시뮬레이션 구간
RHO_B_DB_LIST = (10, 20)                    # Bob 송신 SNR rho_b (dB)
RHO_EDGE_DB_RANGE = np.arange(-20, 21, 5)   # 원 가장자리(가장 약한 위치) Eve의 평균 SNR rho_edge (dB)
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


def d2f_ch4(phi, A, B, D):
    # f''(phi) (4장의 오목성 분석 식)
    return (-A**2 / (1 + A * (1 - phi))**2
            + (D - B)**2 / (1 + B + (D - B) * phi)**2
            - D**2 / (1 + D * phi)**2) / np.log(2.0)


def _newton_in_interval(lo, hi, p0, dfun, d2fun, max_iter=NEWTON_MAX_ITER, tol=NEWTON_TOL):
    # 구간 [lo, hi] 안에서 뉴턴법으로 극대점 탐색 (전제: f'(lo) > 0 > f'(hi), dfun, d2fun: phi -> f', f'')
    # f'(phi) = 0의 근을 뉴턴법과 이분법을 결합해 탐색 (Numerical Recipes의 rtsafe와 같은 구조)
    # 다음 경우 뉴턴법의 갱신값 대신 구간의 중점으로 이동(이분법)
    #   1) 갱신값이 구간을 벗어나거나 f'' >= 0
    #   2) 뉴턴 갱신 폭이 두 반복 전 갱신 폭의 절반보다 큼 (구간이 충분히 빨리 줄지 않음)
    # -> 두 반복마다 갱신 폭이 절반 이하로 줄어 수렴 보장, 해 근처에서는 뉴턴법의 이차 수렴
    lo, hi = lo.astype(float).copy(), hi.astype(float).copy()
    p = np.clip(p0, lo, hi).astype(float)
    n_eval = np.zeros(len(p))
    active = np.ones(len(p), dtype=bool)
    dx_last = hi - lo                                     # 직전 갱신 폭
    dx_before = hi - lo                                   # 두 반복 전 갱신 폭
    for _ in range(max_iter):
        if not active.any():
            break
        g, h = dfun(p), d2fun(p)
        n_eval += active                                  # 반복당 f', f'' 1회 평가
        stationary = np.abs(g) < NEWTON_GTOL              # 이미 정지점이면 그대로 종료
        lo = np.where(active & (g > 0), p, lo)            # 정지점은 f'>0 쪽의 오른쪽
        hi = np.where(active & (g <= 0), p, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_newton = np.where(h < 0, p - g / h, np.nan)
        out = ~((p_newton >= lo) & (p_newton <= hi))       # 구간 이탈 또는 f'' >= 0 (NaN)
        slow = np.abs(p_newton - p) > 0.5 * dx_before      # 갱신 폭이 충분히 줄지 않음
        p_new = np.where(out | slow, 0.5 * (lo + hi), p_newton)
        step = np.abs(p_new - p)
        dx_before = np.where(active, dx_last, dx_before)
        dx_last = np.where(active, step, dx_last)
        done = stationary | (step < tol)
        p = np.where(active & ~stationary, p_new, p)
        active &= ~done
    return p, n_eval


def _interval_around_candidate(grid, k, dfun):
    # 그리드 후보 해 grid[k] 주변 한 칸 선택: 후보 해의 f'이 양수면 오른쪽 칸, 아니면 왼쪽 칸
    #   f'(p_k) > 0 -> [p_k, p_{k+1}],  f'(p_k) <= 0 -> [p_{k-1}, p_k]
    # 반환: lo, hi, 유효 여부(f'(lo) > 0 > f'(hi)), 평가 횟수(2: 후보 해와 이웃점의 기울기)
    last = len(grid) - 1
    p_k = grid[k]
    g_k = dfun(p_k)
    right = g_k > 0
    nb = np.where(right, grid[np.minimum(k + 1, last)], grid[np.maximum(k - 1, 0)])
    g_nb = dfun(nb)
    lo = np.where(right, p_k, nb)
    hi = np.where(right, nb, p_k)
    valid = (hi > lo) & np.where(right, g_nb < 0, g_nb > 0)
    return lo, hi, valid, 2


def optimize_genie_hybrid(A, B, D):
    # 4장 조건부 하이브리드: Eve의 순시 CSI(B, D)를 아는 경우 f(phi)를 채널별로 최대화
    #   안전 구간 (D > B/2): f가 [0,1]에서 오목 -> 양 끝 기울기 확인 후 뉴턴법만 사용
    #   불확실성 구간 (D <= B/2): 낮은 해상도 그리드로 후보 해를 찾고 후보 해 주변 구간에서 뉴턴법으로 정밀화, 후보 해와 비교
    # 반환: phi*, 채널별 평가 횟수(f 또는 f'/f'' 계산 횟수), 안전 구간 여부
    N = len(A)
    phi = np.zeros(N); n_eval = np.zeros(N)
    safe = D > B / 2

    # ----- 안전 구간 -----
    i = np.where(safe)[0]
    if len(i):
        Ai, Bi, Di = A[i], B[i], D[i]
        at0 = df_ch4(np.zeros(len(i)), Ai, Bi, Di) <= 0          # 오목 + f'(0)<=0 -> phi*=0
        atM = ~at0 & (df_ch4(np.full(len(i), PHI_MAX), Ai, Bi, Di) >= 0)  # f'(PHI_MAX)>=0 -> 상한
        n_eval[i] = 1 + (~at0)
        inner = ~at0 & ~atM
        p = np.where(at0, 0.0, PHI_MAX)
        if inner.any():
            k = np.where(inner)[0]
            a, b, d = Ai[k], Bi[k], Di[k]
            pk, nk = _newton_in_interval(np.zeros(len(k)), np.full(len(k), PHI_MAX),
                                         np.full(len(k), NEWTON_INIT),
                                         lambda p: df_ch4(p, a, b, d), lambda p: d2f_ch4(p, a, b, d))
            p[k] = pk; n_eval[i[k]] += nk
        phi[i] = p

    # ----- 불확실성 구간 -----
    j = np.where(~safe)[0]
    if len(j):
        Aj, Bj, Dj = A[j], B[j], D[j]
        low_res = np.linspace(0.0, PHI_MAX, LOW_RES_GRID_N)
        V = f_ch4(low_res[None, :], Aj[:, None], Bj[:, None], Dj[:, None])
        k = np.argmax(V, axis=1)
        p_grid, v_grid = low_res[k], V[np.arange(len(j)), k]
        lo, hi, bracket, n_br = _interval_around_candidate(low_res, k, lambda p: df_ch4(p, Aj, Bj, Dj))
        n_eval[j] = LOW_RES_GRID_N + n_br
        p = p_grid.copy()
        if bracket.any():
            b = np.where(bracket)[0]
            a, bb, d = Aj[b], Bj[b], Dj[b]
            pb, nb = _newton_in_interval(lo[b], hi[b], p_grid[b],
                                         lambda p: df_ch4(p, a, bb, d), lambda p: d2f_ch4(p, a, bb, d))
            better = f_ch4(pb, Aj[b], Bj[b], Dj[b]) >= v_grid[b]
            p[b] = np.where(better, pb, p_grid[b])
            n_eval[j[b]] += nb + 1
        phi[j] = p

    # 보안 전송이 불가능한 채널(max f <= 0)은 인공 잡음을 쓸 이유가 없으므로 phi = 0
    no_secrecy = f_ch4(phi, A, B, D) <= 0
    n_eval += 1
    phi = np.where(no_secrecy, 0.0, phi)
    return phi, n_eval, safe


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


def verify_ch4_hybrid(snr_db_list=tuple(SNR_DB_RANGE), num_samples=20000, ref_points=20001, seed=1):
    # 4장 하이브리드 검증: 초정밀 그리드(ref_points) 대비 손실, 201점 그리드와의 비교, 평가 횟수
    # + 상한 절단 영향: 탐색 구간을 [0, PHI_MAX]로 제한해 [0, 1)의 최적해를 놓치는 채널 비율과 손실
    rng = np.random.default_rng(seed)
    ref = np.linspace(0.0, PHI_MAX, ref_points)
    ext = np.linspace(PHI_MAX, 1.0, int(round((1.0 - PHI_MAX) / (ref[1] - ref[0]))) + 1)[1:]
    ref_full = np.concatenate([ref, ext])                          # [0, PHI_MAX] 기준 그리드 + (PHI_MAX, 1] 같은 간격 확장
    g201 = np.linspace(0.0, PHI_MAX, NUM_PHI)
    print(f"[4장 하이브리드 검증] 표본 {num_samples}, 기준 그리드 {ref_points}점 (상한 절단 확인용 확장 {len(ext)}점)")
    print(f"{'SNR(dB)':>7} | {'안전구간':>8} | {'손실 평균':>10} | {'손실 최대':>10} | "
          f"{'201점 최대손실':>13} | {'평가횟수 평균':>12} | {'최대':>4} | "
          f"{'절단 채널(%)':>11} | {'절단 최대손실':>12} | {'절단 평균손실':>12}")
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        A, B, D = generate_gains(rng, num_samples, rho, rho)
        phi, n_eval, safe = optimize_genie_hybrid(A, B, D)
        v_h = np.maximum(0, f_ch4(phi, A, B, D))
        v_ref = np.empty(num_samples); v_201 = np.empty(num_samples); v_full = np.empty(num_samples)
        for s0 in range(0, num_samples, 1000):                     # 메모리 절약용 청크
            sl = slice(s0, s0 + 1000)
            a, b, d = A[sl, None], B[sl, None], D[sl, None]
            F = np.maximum(0, f_ch4(ref_full[None, :], a, b, d))
            v_full[sl] = F.max(1)
            v_ref[sl] = F[:, :ref_points].max(1)                    # [0, PHI_MAX] 기준값
            v_201[sl] = np.maximum(0, f_ch4(g201[None, :], a, b, d)).max(1)
        loss = v_ref - v_h
        trunc = v_full - v_ref                                      # 상한 절단 손실 (>= 0, 기준값끼리 비교 -> 알고리즘 오차와 분리)
        cut = trunc > 0                                             # [0, 1)의 최적해가 (PHI_MAX, 1)에 놓인 채널
        print(f"{snr_db:>7} | {safe.mean():>8.3f} | {loss.mean():>10.1e} | {loss.max():>10.1e} | "
              f"{(v_ref - v_201).max():>13.1e} | {n_eval.mean():>12.1f} | {int(n_eval.max()):>4} | "
              f"{cut.mean() * 100:>11.3f} | {trunc.max():>12.1e} | {trunc.mean():>12.1e}")


# ====================================
# 5장: Eve 위치 불확실성 (원 모델)
# - Eve는 Alice 중심 반지름 R인 원 안 어딘가에 면적 균일하게 위치, 위치는 전송 동안 고정이며 Alice는 모름
#   -> rho_e = rho_edge * (r/R)^-alpha, rho_edge: 원 가장자리(가장 약한 위치) Eve의 평균 SNR
# - 블록마다 Alice는 h를 알고 phi, R_b(=Cb), R_s를 정함. 제약: 위치·g 평균 SOP <= eps
# - 성능: 보안 처리량 E_h[R_s] (Eve 순시 CSI가 있으면 E[Cs]가 되므로 4장 Genie가 상한)
# ====================================


def disk_L(s, rho_edge, alpha=PL_ALPHA):
    # 위치 항 L(s) = E_r[exp(-s / rho_e(r))]
    # 닫힌형: L(s) = d (rho_edge/s)^d * lower_gamma(d, s/rho_edge), d = 2/alpha (s -> 0이면 1)
    d = 2.0 / alpha
    x = np.maximum(np.asarray(s, dtype=float) / rho_edge, 1e-300)
    return d * x ** (-d) * gammainc(d, x) * gamma_fn(d)


def disk_dL(s, rho_edge, alpha=PL_ALPHA):
    # L'(s) = -E[(1/rho_e) exp(-s/rho_e)] = -(d/rho_edge) x^(-d-1) lower_gamma(d+1, x), x = s/rho_edge
    d = 2.0 / alpha
    x = np.maximum(np.asarray(s, dtype=float) / rho_edge, 1e-300)
    return -(d / rho_edge) * x ** (-d - 1) * gammainc(d + 1, x) * gamma_fn(d + 1)


def eve_ccdf_disk(t, phi, rho_edge, alpha=PL_ALPHA):
    # 위치·g 평균 Eve SINR 꼬리확률 P(gamma_e > t) = L(t/(1-phi)) / (1 + t phi/(1-phi))  (Nt=2)
    # 분모(AN 항)는 위치와 무관, 위치 정보는 L에만 들어감
    return disk_L(t / (1 - phi), rho_edge, alpha) / (1 + t * phi / (1 - phi))


def sop_quantile(phi_grid, eps, L_fun, iters=100):
    # q_eps(phi): P(gamma_e > q) = eps인 분위수, q = (1-phi) s
    # s는 L(s) = eps (1 + phi s)의 유일한 근 (좌변 감소, 우변 증가) -> log s 이분법 (phi 격자 전체 벡터화)
    # L_fun: 위치 항 L(s) = E[exp(-s/rho_e)] (rho_e의 분포만 바꾸면 그대로 적용)
    phi = np.asarray(phi_grid, dtype=float)
    lo = np.full(phi.shape, -60.0)
    hi = np.full(phi.shape, 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = np.exp(mid)
        left = L_fun(s) > eps * (1 + phi * s)                       # 근보다 왼쪽
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    return (1 - phi) * np.exp(0.5 * (lo + hi))


def sop_quantile_disk(phi_grid, eps, rho_edge, alpha=PL_ALPHA):
    # 원 모델의 분위수
    return sop_quantile(phi_grid, eps, lambda s: disk_L(s, rho_edge, alpha))


def sop_quantile_fixed(phi_grid, eps, rho_e):
    # rho_e를 값 하나로 볼 때의 분위수 (L(s) = exp(-s/rho_e))
    return sop_quantile(phi_grid, eps, lambda s: np.exp(-s / rho_e))


def sop_quantile_noiseless(phi_grid, eps):
    # Eve 수신 잡음 0 (rho_e -> inf): P(gamma_e > t) = 1/(1 + t phi/(1-phi)) -> q = (1-phi)(1-eps)/(phi eps)
    # 원 안 모든 위치에 대해 SOP <= eps를 보장하는 분위수 (rho_e가 클수록 Eve가 강하므로 상한)
    phi = np.asarray(phi_grid, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(phi > 0, (1 - phi) * (1 - eps) / (phi * eps), np.inf)   # phi = 0이면 보안 전송 불가


def harmonic_mean_disk(rho_edge, alpha=PL_ALPHA):
    # 원 모델에서 rho_e의 조화평균 1/E[1/rho_e] = (1 + alpha/2) rho_edge (산술평균은 발산)
    # rho_eff(s)의 s -> 0 극한
    return (1 + alpha / 2) * rho_edge


def rho_eff_disk(s, rho_edge, alpha=PL_ALPHA):
    # 등가 SNR rho_eff(s) = -s / ln L(s): 같은 s에서 위치 항 L(s)와 같은 값을 주는 단일 rho_e
    # 분위수 방정식 L(s) = eps(1 + phi s)에서 L(s) = exp(-s/rho_eff)로 바꿔도 근이 같음
    # -> q_eps(phi)는 rho_e = rho_eff(s(phi))인 단일 SNR의 분위수와 정확히 같음
    # ln L(s)가 s에 대해 볼록(누적생성함수) -> rho_eff(s)는 s에 대해 증가, s -> 0이면 조화평균, s -> inf이면 rho_max
    return -np.asarray(s, dtype=float) / np.log(disk_L(s, rho_edge, alpha))


def an_on_threshold_disk(eps, rho_edge, alpha=PL_ALPHA):
    # AN 사용 조건: A/(1+A) < T 이면 phi* > 0 (R_s*(phi)의 phi = 0에서의 기울기 > 0)
    #   T = s0 (1 + eps/|L'(s0)|) / (1 + s0),  s0 = L^{-1}(eps) (phi = 0의 분위수)
    # T >= 1이면 모든 A에서 AN 사용
    s0 = sop_quantile_disk(np.array([0.0]), eps, rho_edge, alpha)[0]           # phi = 0이면 q = s
    return s0 * (1 + eps / abs(disk_dL(s0, rho_edge, alpha))) / (1 + s0)


def generate_gains_disk(rng, n, rho_b, rho_edge, alpha=PL_ALPHA):
    # 표본마다 Eve 위치를 원 안 면적 균일로 생성: (r/R)^2 = u ~ U(0, 1] -> rho_e = rho_edge * u^(-alpha/2)
    u = 1.0 - rng.uniform(size=n)                                   # (0, 1] (u = 0 제외)
    rho_e = rho_edge * u ** (-alpha / 2)
    A, B, D = generate_gains(rng, n, rho_b, rho_e)                  # B, D는 표본별 rho_e로 계산 (식 3.8)
    return A, B, D, rho_e


def optimize_ch5_disk(A, phi_fine, q_fine, coarse_step=CH5_COARSE_STEP, chunk=20000):
    # 블록별 R_s(phi) = log2(1 + A(1-phi)) - log2(1 + q_eps(phi)) 최대화 (q_eps는 h와 무관 -> 사전 계산해 공유)
    # R_s(phi)의 내부 극대점은 하나 이하(수치 확인) -> 거친 격자 argmax 후 이웃 구간에서 정밀 격자 argmax
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
    # 정해진 phi(블록별 또는 고정)에서 실제 분위수로 정한 R_s = [log2(1 + A(1-phi)) - log2(1 + q_eps(phi))]^+
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


def verify_ch5_disk(rho_b_db_list=(0, 10, 20), rho_edge_db_list=(-10, 0, 10), eps_list=EPS_LIST,
                    num_samples=200000, seed=3, alpha=PL_ALPHA):
    # 5장 해석식 검증
    #   A. 위치 항 L(s): 닫힌형 vs 수치적분 (Gauss-Legendre)
    #   B. 위치·g 평균 Eve SINR 꼬리확률: 해석식 vs MC (실제 h, g, 위치 표본)
    #   C. 설계한 (phi*, R_b, R_s)에서 전송 블록의 실제 누설 비율(MC)이 eps와 같은지
    rng = np.random.default_rng(seed)
    glx, glw = np.polynomial.legendre.leggauss(4000)
    u, w = (glx + 1) / 2, glw / 2
    print(f"[5장 해석식 검증 A] 위치 항 L(s): 닫힌형 vs 수치적분 (alpha = {alpha})")
    print(f"{'rho_edge(dB)':>12} | {'최대 상대오차':>12}")
    for re_db in rho_edge_db_list:
        re = 10 ** (re_db / 10)
        ss = np.geomspace(1e-3, 1e3, 25) * re
        num = np.array([np.sum(w * np.exp(-s * u ** (alpha / 2) / re)) for s in ss])
        print(f"{re_db:>12} | {np.max(np.abs(disk_L(ss, re, alpha) / num - 1)):>12.1e}")

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

    print(f"\n[5장 해석식 검증 C] 설계한 (phi*, R_b, R_s)에서 전송 블록의 누설 비율 (표본 {num_samples})")
    print(f"{'rho_b(dB)':>9} | {'rho_edge(dB)':>12} | {'eps':>5} | {'전송 비율':>9} | {'누설 비율 MC':>11} | {'z':>5} | {'보안 처리량':>10}")
    phi_fine = np.linspace(0.0, PHI_MAX, CH5_PHI_FINE_N)
    for rb_db in rho_b_db_list:
        rho_b = 10 ** (rb_db / 10)
        for re_db in rho_edge_db_list:
            re = 10 ** (re_db / 10)
            A, B, D, _ = generate_gains_disk(rng, num_samples, rho_b, re, alpha)
            for eps in eps_list:
                q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
                phi_opt, rs = optimize_ch5_disk(A, phi_fine, q_fine)
                tx = rs > 0
                q_opt = np.interp(phi_opt, phi_fine, q_fine)          # phi_opt는 격자점 -> 정확히 q_eps(phi*)
                leak = (B * (1 - phi_opt) / (1 + D * phi_opt) > q_opt)[tx]   # Ce > R_b - R_s  <=>  gamma_e > q_eps(phi*)
                if tx.sum() == 0:                                      # 전송 블록 없음 -> 누설 비율 정의되지 않음
                    print(f"{rb_db:>9} | {re_db:>12} | {eps:>5.2f} | {tx.mean():>9.3f} | {'-':>11} | {'-':>5} | {rs.mean():>10.4f}")
                    continue
                p_leak = leak.mean()
                z = abs(p_leak - eps) / np.sqrt(eps * (1 - eps) / tx.sum())
                print(f"{rb_db:>9} | {re_db:>12} | {eps:>5.2f} | {tx.mean():>9.3f} | {p_leak:>11.5f} | {z:>5.2f} | {rs.mean():>10.4f}")

    print(f"\n[5장 해석식 검증 D] 블록별 탐색(거친 격자 + 이웃 정밀 격자) vs 정밀 격자 전수 탐색 ({CH5_PHI_FINE_N}점)")
    print(f"{'rho_b(dB)':>9} | {'rho_edge(dB)':>12} | {'eps':>5} | {'손실 최대':>10}")
    n_d = 20000
    for rb_db in rho_b_db_list:
        rho_b = 10 ** (rb_db / 10)
        A = rho_b * rng.gamma(NT, 1.0, size=n_d)                       # ||h||^2 ~ Gamma(Nt, 1)
        for re_db in rho_edge_db_list:
            re = 10 ** (re_db / 10)
            for eps in eps_list:
                q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
                _, rs = optimize_ch5_disk(A, phi_fine, q_fine)
                full = np.empty(n_d)
                for s0 in range(0, n_d, 2000):                         # 메모리 절약용 청크
                    a = A[s0:s0 + 2000, None]
                    full[s0:s0 + 2000] = np.maximum(0.0, (np.log2(1 + a * (1 - phi_fine[None, :])) - np.log2(1 + q_fine)[None, :]).max(1))
                print(f"{rb_db:>9} | {re_db:>12} | {eps:>5.2f} | {(full - rs).max():>10.1e}")

    print(f"\n[5장 해석식 검증 E] AN 사용 조건 A/(1+A) < T: 닫힌형 판정 vs 정밀 격자 전수 탐색 (A = -10 ~ 40 dB, 0.5 dB 간격)")
    print(f"{'rho_edge(dB)':>12} | {'eps':>5} | {'T':>9} | {'AN 사용 A 구간':>16} | {'불일치':>6}")
    A_sweep = 10 ** (np.arange(-10, 40.01, 0.5) / 10)
    for re_db in (-20, -15) + tuple(rho_edge_db_list):
        re = 10 ** (re_db / 10)
        for eps in eps_list:
            T = an_on_threshold_disk(eps, re, alpha)
            q_fine = sop_quantile_disk(phi_fine, eps, re, alpha)
            R = np.log2(1 + A_sweep[:, None] * (1 - phi_fine[None, :])) - np.log2(1 + q_fine)[None, :]
            brute_on = R.max(1) > R[:, 0] + 1e-12                              # phi > 0에서 더 큰 값이 있음
            pred_on = A_sweep / (1 + A_sweep) < T
            rng_txt = "모든 A" if T >= 1 else f"A < {10 * np.log10(T / (1 - T)):.2f} dB"
            print(f"{re_db:>12} | {eps:>5.2f} | {T:>9.3f} | {rng_txt:>16} | {int(np.sum(brute_on != pred_on)):>3}/{len(A_sweep)}")

    print(f"\n[5장 해석식 검증 F] 등가 SNR rho_eff: 분위수 등가성, 단조성, 조화평균 극한")
    print(f"{'rho_edge(dB)':>12} | {'eps':>5} | {'q 등가 최대 상대오차':>18} | {'단조 증가':>8} | {'rho_eff(s->0)/조화평균':>21}")
    s_scan = np.geomspace(1e-6, 1e6, 200)
    for re_db in rho_edge_db_list:
        re = 10 ** (re_db / 10)
        r_eff = rho_eff_disk(s_scan * re, re, alpha)
        mono = bool(np.all(np.diff(r_eff) > 0))
        lim = r_eff[0] / harmonic_mean_disk(re, alpha)
        for eps in eps_list:
            phis = np.linspace(0.0, 0.95, 20)
            q_disk = sop_quantile_disk(phis, eps, re, alpha)
            s_phi = q_disk / (1 - phis)
            q_equiv = np.array([sop_quantile_fixed(np.array([p]), eps, rho_eff_disk(s, re, alpha))[0] for p, s in zip(phis, s_phi)])
            print(f"{re_db:>12} | {eps:>5.2f} | {np.max(np.abs(q_equiv / q_disk - 1)):>18.1e} | {str(mono):>8} | {lim:>21.6f}")


# ====================================
# 6장: 원 모델에서의 기법 비교
# ====================================
SCHEMES = ("genie", "proposed", "noiseless_phi", "noiseless", "static")
PLOT_SCHEMES = ("genie", "proposed", "noiseless_phi", "noiseless")     # Static은 Proposed와 겹쳐 표로만 출력
SCHEME_STYLE = {   # 라벨, 색, 선·마커 (기법별 통일)
    "genie":         ("Genie-aided (Eve CSI)", "#10b981", "^-."),
    "proposed":      ("Proposed", "#3b82f6", "s-"),
    "noiseless_phi": ("Noiseless-Eve phi + true R_s", "#f59e0b", "o--"),
    "noiseless":     ("Noiseless-Eve design (all positions)", "#ef4444", "v--"),
    "static":        ("Static (fixed phi)", "#8b5cf6", "x:"),
}


def run_simulation(num_samples=NUM_SAMPLES, rho_b_db_list=RHO_B_DB_LIST, rho_edge_db_range=RHO_EDGE_DB_RANGE,
                   eps_list=EPS_LIST, seed=SEED, alpha=PL_ALPHA):
    # 원 모델에서 기법별 보안 처리량, 전송 블록의 평균 phi, 전송 블록의 실제 누설 비율(MC)
    # (rho_b, rho_edge)마다 채널·위치 표본을 한 번 생성해 eps와 모든 기법이 공유
    rng = np.random.default_rng(seed)
    phi_fine = np.linspace(0.0, PHI_MAX, CH5_PHI_FINE_N)
    q_noiseless = {eps: sop_quantile_noiseless(phi_fine, eps) for eps in eps_list}  # rho_edge와 무관
    res = {(rb, eps): {"rho_edge": [], **{f"{m}_{k}": [] for m in ("eta", "phi", "leak") for k in SCHEMES}}
           for rb in rho_b_db_list for eps in eps_list}

    print(f"시뮬레이션 시작 (Samples: {num_samples}, alpha: {alpha})")
    for rb_db in rho_b_db_list:
        rho_b = 10 ** (rb_db / 10)
        for re_db in rho_edge_db_range:
            re = 10 ** (re_db / 10)
            A, B, D, _ = generate_gains_disk(rng, num_samples, rho_b, re, alpha)

            # Genie: Eve 순시 CSI(위치 포함)로 블록마다 4장 알고리즘 -> R_s = [Cb - Ce]^+ (eps와 무관, 누설 없음)
            phi_g, _, _ = optimize_genie_hybrid(A, B, D)
            rs_g = np.maximum(0.0, f_ch4(phi_g, A, B, D))

            for eps in eps_list:
                q_true = sop_quantile_disk(phi_fine, eps, re, alpha)                    # 실제 위치 분포의 분위수
                phi_n, rs_n = optimize_ch5_disk(A, phi_fine, q_noiseless[eps])          # Eve 수신 잡음 0 가정 설계
                phi_s = optimize_static_ch5(rho_b, phi_fine, q_true)
                out = {
                    "genie":         (phi_g, rs_g),
                    "proposed":      optimize_ch5_disk(A, phi_fine, q_true),
                    "noiseless_phi": (phi_n, rs_at_phi(A, phi_n, phi_fine, q_true)),    # phi는 수신 잡음 0 가정, R_s는 실제 분포
                    "noiseless":     (phi_n, rs_n),                                     # phi, R_s 모두 수신 잡음 0 가정
                    "static":        (np.full(num_samples, phi_s), rs_at_phi(A, phi_s, phi_fine, q_true)),
                }
                r = res[(rb_db, eps)]
                r["rho_edge"].append(re_db)
                for k, (phi_k, rs_k) in out.items():
                    tx = rs_k > 0
                    r[f"eta_{k}"].append(rs_k.mean())
                    r[f"phi_{k}"].append(phi_k[tx].mean() if tx.any() else np.nan)
                    if k == "genie" or not tx.any():
                        r[f"leak_{k}"].append(0.0 if k == "genie" else np.nan)
                        continue
                    thr = (1 + A * (1 - phi_k)) / 2 ** rs_k - 1                        # 2^(R_b - R_s) - 1
                    r[f"leak_{k}"].append(np.mean((B * (1 - phi_k) / (1 + D * phi_k) > thr)[tx]))

    # ----- 결과 표 출력 -----
    for (rb_db, eps), r in res.items():
        print(f"\n[rho_b = {rb_db} dB, eps = {eps}] 보안 처리량 (bps/Hz), 전송 블록 평균 phi, 전송 블록의 실제 누설 비율")
        print(f"{'rho_edge':>8} | " + " | ".join(f"{k:>13}" for k in SCHEMES)
              + f" | {'phi(P)':>6} | {'phi(N)':>6} | {'누설(P)':>7} | {'누설(N-phi)':>9} | {'누설(N)':>7}")
        for i, re_db in enumerate(r["rho_edge"]):
            print(f"{re_db:>8} | " + " | ".join(f"{r[f'eta_{k}'][i]:>13.4f}" for k in SCHEMES)
                  + f" | {r['phi_proposed'][i]:>6.3f} | {r['phi_noiseless'][i]:>6.3f} | {r['leak_proposed'][i]:>7.4f}"
                  + f" | {r['leak_noiseless_phi'][i]:>9.4f} | {r['leak_noiseless'][i]:>7.4f}")
        gap = 100 * (1 - np.array(r["eta_static"]) / np.array(r["eta_proposed"]))
        print(f"  Static의 Proposed 대비 손실: 최대 {np.nanmax(gap):.2f}% (phi 블록 적응의 이득)")
    return res


def plot_results(res, rho_b_db_list=RHO_B_DB_LIST, eps_list=EPS_LIST):
    nr, nc = len(rho_b_db_list), len(eps_list)

    # ===== (1) rho_edge에 따른 보안 처리량 =====
    plt.figure(figsize=(6.5 * nc, 4.8 * nr))
    for i, rb_db in enumerate(rho_b_db_list):
        for j, eps in enumerate(eps_list):
            r = res[(rb_db, eps)]
            plt.subplot(nr, nc, i * nc + j + 1)
            for k in PLOT_SCHEMES:
                label, color, style = SCHEME_STYLE[k]
                plt.plot(r["rho_edge"], r[f"eta_{k}"], style, color=color, label=label, linewidth=1.6,
                         markersize=6, markerfacecolor="none", markeredgewidth=1.4)
            plt.xlabel("Eve SNR at disk edge, rho_edge (dB)")
            plt.ylabel("Secrecy Throughput (bps/Hz)")
            plt.title(f"rho_b = {rb_db} dB, eps = {eps}")
            plt.grid(True, linestyle=":", alpha=0.7)
            if i == 0 and j == 0:
                plt.legend(fontsize=8)
    plt.tight_layout()

    # ===== (2) rho_edge에 따른 평균 phi (전송 블록 기준) =====
    plt.figure(figsize=(6.5 * nc, 4.8 * nr))
    for i, rb_db in enumerate(rho_b_db_list):
        for j, eps in enumerate(eps_list):
            r = res[(rb_db, eps)]
            plt.subplot(nr, nc, i * nc + j + 1)
            for k in ("genie", "proposed", "noiseless"):
                label, color, style = SCHEME_STYLE[k]
                plt.plot(r["rho_edge"], r[f"phi_{k}"], style, color=color, label=label, linewidth=1.6,
                         markersize=6, markerfacecolor="none", markeredgewidth=1.4)
            plt.xlabel("Eve SNR at disk edge, rho_edge (dB)")
            plt.ylabel("Average phi (transmitted blocks)")
            plt.title(f"rho_b = {rb_db} dB, eps = {eps}")
            plt.ylim(0, 1)
            plt.grid(True, linestyle=":", alpha=0.7)
            if i == 0 and j == 0:
                plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 검증 (RUN_VERIFY = True일 때만)
    if RUN_VERIFY:
        verify_ch4_hybrid()
        verify_ch5_disk()

    # 시뮬레이션 수행
    plot_results(run_simulation())
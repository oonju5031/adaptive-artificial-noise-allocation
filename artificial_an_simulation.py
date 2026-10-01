import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1

# ====================================
# 적응형 AN 전력 할당 시뮬레이션
#
# - 시스템
#   - MISO Wiretap (Alice Nt=2, Bob 1, Eve 1) -> TODO: 이후 N_t > 2인 경우로 확장 예정
# - 비교
#   - Fixed(전력비 고정 기법)
#   - Genie-aided(Eve의 CSI를 아는 경우, 성능 상한선)
#   - Proposed(Eve의 CSI를 모르는 경우)
#   - SNR-static(SNR마다 채널 통계로 정한 고정 phi, h 순시값에 따른 적응 없음)
# - 평가 지표
#   - 에르고딕 보안 용량: Genie는 E[[Cb-Ce]^+] (Eve CSI로 블록별 판정 가능)
#     Fixed, Proposed는 달성 가능 전송률 E[1{전송}(Cb-Ce)] (h만으로 전송 여부 결정, 누설 블록 포함)
#   - SOP: 블록별 P(Cb - Ce < Rs) (5.6절 닫힌형과 같은 정의)
# ====================================


# ----- HyperParameters --------------
NUM_SAMPLES = 50000                     # Monte-Carlo 채널 표본 수
RS = 1.0                                # 목표 보안 전송률 (bps/Hz)
SIGMA_B2 = 1.0                          # Bob 잡음 분산 (모의실험 설정: Eve와 동일)
SIGMA_E2 = 1.0                          # Eve 잡음 분산
NUM_PHI = 201                           # Grid search 해상도(정밀도)
SEED = 0                                # 난수 시드 (재현성)
SNR_DB_RANGE = np.arange(-20, 31, 5)    # 송신 SNR 구간 (dB)
RUN_VERIFY = False                      # True면 시뮬레이션 전에 검증 함수 실행 (터미널에 검증 수치 출력)

NT = 2              # 송신 안테나 수 (현 공식은 2로 고정된 경우에 한정됨, TODO: 이후 N_t > 2인 경우로 확장 예정)

FIXED_PHI = 0.5     # 전력비 고정 기법의 전력비
STATIC_TRAIN_N = 200000   # SNR-static의 phi 계산용 독립 표본 수 (시뮬레이션 표본과 별도)
STATIC_SEED = 12345       # SNR-static 표본 시드 (시뮬레이션 시드와 분리)
PHI_MAX = 0.99      # phi 탐색 상한 (1.0인 경우 신호 전력이 0이므로 제외)

EG_QUAD_N = 400     # E_g[Ce]의 Y(AN 성분) 수치적분 격자점 수
EG_QUAD_YMAX = 60.0 # Y 적분 상한 (Exp(1) tail, e^{-60}은 무시 가능)

# 뉴턴법 (4장, 5장 공통, 구간을 벗어나면 이분법으로 보완)
NEWTON_INIT = 0.3       # 뉴턴법 초기값
NEWTON_MAX_ITER = 50    # 최대 반복 수 (이분법 대체 포함)
NEWTON_TOL = 1e-9       # 수렴 판정 (phi 변화량)
NEWTON_GTOL = 1e-12     # 수렴 판정 (|f'| 크기)

# 낮은 해상도 그리드 (4장 불확실성 구간, 5장 그리드 탐색이 필요한 경우 공통)
LOW_RES_GRID_N = 11     # 낮은 해상도 그리드 점 수 (0~PHI_MAX 균등 분할)
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


def _stable_exp_e1(z):
    # e^z * E1(z). z가 크면 exp 오버플로 -> 점근급수로 대체
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = z < 30
    out[small] = np.exp(z[small]) * exp1(z[small])
    zl = z[~small]
    out[~small] = (1/zl) * (1 - 1/zl + 2/zl**2 - 6/zl**3 + 24/zl**4)
    return out


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
    # 뉴턴 스텝이 구간을 벗어나거나 f'' >= 0이면 구간 중앙으로 이동(이분법) -> 구간이 매 반복 줄어 수렴 보장
    lo, hi = lo.astype(float).copy(), hi.astype(float).copy()
    p = np.clip(p0, lo, hi).astype(float)
    n_eval = np.zeros(len(p))
    active = np.ones(len(p), dtype=bool)
    for _ in range(max_iter):
        if not active.any():
            break
        g, h = dfun(p), d2fun(p)
        n_eval += active                                  # 반복당 f', f'' 1회 평가
        stationary = np.abs(g) < NEWTON_GTOL              # 이미 정지점이면 그대로 종료
        lo = np.where(active & (g > 0), p, lo)            # 정지점은 f'>0 쪽의 오른쪽
        hi = np.where(active & (g <= 0), p, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_new = np.where(h < 0, p - g / h, np.nan)
        bad = ~((p_new >= lo) & (p_new <= hi))            # 경계 포함 판정 (NaN은 bad)
        p_new = np.where(bad, 0.5 * (lo + hi), p_new)
        done = stationary | (np.abs(p_new - p) < tol)
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
    rng = np.random.default_rng(seed)
    ref = np.linspace(0.0, PHI_MAX, ref_points)
    g201 = np.linspace(0.0, PHI_MAX, NUM_PHI)
    print(f"[4장 하이브리드 검증] 표본 {num_samples}, 기준 그리드 {ref_points}점")
    print(f"{'SNR(dB)':>7} | {'안전구간':>8} | {'손실 평균':>10} | {'손실 최대':>10} | "
          f"{'201점 최대손실':>13} | {'평가횟수 평균':>12} | {'최대':>4}")
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        A, B, D = generate_gains(rng, num_samples, rho, rho)
        phi, n_eval, safe = optimize_genie_hybrid(A, B, D)
        v_h = np.maximum(0, f_ch4(phi, A, B, D))
        v_ref = np.empty(num_samples); v_201 = np.empty(num_samples)
        for s0 in range(0, num_samples, 1000):                     # 메모리 절약용 청크
            sl = slice(s0, s0 + 1000)
            a, b, d = A[sl, None], B[sl, None], D[sl, None]
            v_ref[sl] = np.maximum(0, f_ch4(ref[None, :], a, b, d)).max(1)
            v_201[sl] = np.maximum(0, f_ch4(g201[None, :], a, b, d)).max(1)
        loss = v_ref - v_h
        print(f"{snr_db:>7} | {safe.mean():>8.3f} | {loss.mean():>10.1e} | {loss.max():>10.1e} | "
              f"{(v_ref - v_201).max():>13.1e} | {n_eval.mean():>12.1f} | {int(n_eval.max()):>4}")


def ergodic_E_Ce(phi_grid, rho_e, derivs=False):
    # E_g[Ce](phi) 를 phi_grid 전체에 대해 계산 (bps/Hz). rho_e = P_total / sigma_e^2
    # Nt = 2에서 Eve의 신호 / AN 이득이 지수분포를 따름을 이용해 신호 성분은 지수적분(E1)으로 해석적으로 처리하고 AN 성분만 수치적분
    # h와 무관하므로 SNR당 1회만 계산
    # derivs=True 이면 phi에 대한 1·2계 해석적 도함수도 함께 반환 (뉴턴법용)
    # 고 SNR에서 피적분함수가 y=0 근처에서 급변하므로 원점 근처를 촘촘히 배치 (비균일 격자)
    y = np.concatenate([[0.0], np.geomspace(1e-6, EG_QUAD_YMAX, EG_QUAD_N - 1)])[None, :]  # [1, Qy]
    wy = np.exp(-y)                                             # Exp(1) 가중치
    one_minus = (1.0 - phi_grid)[:, None]                       # [G, 1]
    c = rho_e * one_minus / (1.0 + rho_e * phi_grid[:, None] * y)   # [G, Qy]
    c = np.maximum(c, 1e-15)
    Fu = _stable_exp_e1(1.0 / c)                               # F(u)=e^u E1(u), u=1/c [G,Qy]
    E_nat = np.trapezoid(Fu * wy, y[0], axis=1)                # [G]
    E_Ce = E_nat / np.log(2.0)                                 # nat -> bps/Hz
    if not derivs:
        return E_Ce

    # phi 도함수
    u = 1.0 / c                                         # [G,Qy]
    up = (1.0 + rho_e * y) / (rho_e * one_minus**2)     # u'(phi)  [G,Qy]
    upp = 2.0 * (1.0 + rho_e * y) / (rho_e * one_minus**3)  # u''(phi) [G,Qy]
    Fp = Fu - 1.0 / u                                   # F'(u)
    Fpp = Fu - 1.0 / u + 1.0 / u**2                     # F''(u)
    dE = np.trapezoid((Fp * up) * wy, y[0], axis=1) / np.log(2.0)                   # E_Ce'(phi) [G]
    d2E = np.trapezoid((Fpp * up**2 + Fp * upp) * wy, y[0], axis=1) / np.log(2.0)   # E_Ce''(phi) [G]
    return E_Ce, dE, d2E


def an_off_threshold(rho_e):
    # 인공 잡음 미사용 조건의 임계값 T(rho_e) = (1 + rho_e)(1 - e^{1/rho_e} E1(1/rho_e) / rho_e)
    # C_bar'(0) <= 0  <=>  A / (1 + A) >= T(rho_e)  이면 phi* = 0
    return (1.0 + rho_e) * (1.0 - _stable_exp_e1(np.array([1.0 / rho_e]))[0] / rho_e)


def optimize_proposed_hybrid(A, rho_e, phi_grid, E_Ce, dE_Ce, d2E_Ce):
    # 5장 목적함수 C_bar(phi) = Cb(phi|h) - E_g[Ce](phi) 를 채널별로 최대화 (최종 전송률은 [.]^+)
    #   (조건부 에르고딕 보안 용량 E_g[[Cb-Ce]^+ | h]의 하한)
    # E_g[Ce]와 도함수는 SNR당 1회 phi_grid 위에서 계산해 모든 채널이 공유 -> 선형보간으로 연속 평가
    # 절차 (5.4절: 비오목 구간은 phi*의 오른쪽에만 존재)
    #   1) phi*=0 판정: 닫힌형 A/(1+A) >= T(rho_e) 이면 phi* = 0
    #   2) 그 외: [0, PHI_MAX]에서 뉴턴법으로 극대점 탐색
    #   3) 그리드 탐색이 필요한 경우(상한에서 기울기 >= 0, 또는 찾은 점에서 f'' >= 0):
    #      낮은 해상도 그리드 + 후보 해 주변 구간 뉴턴법 (저 SNR에서는 대부분 보안 전송이 불가능한 채널)
    # 반환: phi*, 채널별 평가 횟수, phi*=0 판정 여부, 그리드 탐색 필요 여부
    ln2 = np.log(2.0); N = len(A)
    phi = np.zeros(N); n_eval = np.ones(N)                 # phi*=0 판정 1회
    off = A / (1.0 + A) >= an_off_threshold(rho_e)
    needs_grid = np.zeros(N, dtype=bool)

    def C(p, a):
        return np.log2(1 + a * (1 - p)) - np.interp(p, phi_grid, E_Ce)

    def dC(p, a):
        return (-a / (1 + a * (1 - p))) / ln2 - np.interp(p, phi_grid, dE_Ce)

    def d2C(p, a):
        return (-a**2 / (1 + a * (1 - p))**2) / ln2 - np.interp(p, phi_grid, d2E_Ce)

    k = np.where(~off)[0]
    if len(k):
        a = A[k]
        upper_ok = dC(np.full(len(k), PHI_MAX), a) < 0     # 상한에서 기울기 < 0 이어야 구간 [0, PHI_MAX] 성립
        n_eval[k] += 1
        pk = np.zeros(len(k)); ng = ~upper_ok
        u = np.where(upper_ok)[0]
        if len(u):
            au = a[u]
            pu, nu = _newton_in_interval(np.zeros(len(u)), np.full(len(u), PHI_MAX), np.full(len(u), NEWTON_INIT),
                                         lambda p: dC(p, au), lambda p: d2C(p, au))
            n_eval[k[u]] += nu
            pk[u] = pu
            ng[u] = d2C(pu, au) >= 0                        # 극대가 아니면 그리드 탐색
        if ng.any():                                        # 그리드 탐색: 낮은 해상도 그리드 + 후보 해 주변 구간 뉴턴법 (4장 불확실성 구간과 동일)
            f_idx = np.where(ng)[0]; af = a[f_idx]
            low_res = np.linspace(0.0, PHI_MAX, LOW_RES_GRID_N)
            V = C(low_res[None, :], af[:, None])
            j = np.argmax(V, axis=1)
            p_c, v_c = low_res[j], V[np.arange(len(f_idx)), j]
            lo, hi, br, n_br = _interval_around_candidate(low_res, j, lambda p: dC(p, af))
            n_eval[k[f_idx]] += LOW_RES_GRID_N + n_br
            if br.any():
                b = np.where(br)[0]; ab = af[b]
                pb, nb = _newton_in_interval(lo[b], hi[b], p_c[b], lambda p: dC(p, ab), lambda p: d2C(p, ab))
                p_c[b] = np.where(C(pb, ab) >= v_c[b], pb, p_c[b])
                n_eval[k[f_idx[b]]] += nb + 1
            pk[f_idx] = p_c
        phi[k] = pk; needs_grid[k] = ng
    # 보안 전송이 불가능한 채널(max C_bar <= 0)은 인공 잡음을 쓸 이유가 없으므로 phi = 0
    no_secrecy = C(phi, A) <= 0
    n_eval += 1
    phi = np.where(no_secrecy, 0.0, phi)
    return phi, n_eval, off, needs_grid


def verify_ch5_hybrid(snr_db_list=tuple(SNR_DB_RANGE), num_samples=20000, ref_points=20001, seed=2):
    # 5장 하이브리드 검증: E_g[Ce]를 초정밀 그리드(ref_points)에서 직접 계산한 참값 기준 목적함수 손실, 평가 횟수
    rng = np.random.default_rng(seed)
    ref = np.linspace(0.0, PHI_MAX, ref_points)
    g201 = np.linspace(0.0, PHI_MAX, NUM_PHI)
    print(f"[5장 하이브리드 검증] 표본 {num_samples}, 기준 그리드 {ref_points}점")
    print(f"{'SNR(dB)':>7} | {'phi*=0':>7} | {'그리드':>7} | {'손실 평균':>10} | {'손실 최대':>10} | "
          f"{'201점 최대손실':>13} | {'평가횟수 평균':>12} | {'최대':>4}")
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        A, _, _ = generate_gains(rng, num_samples, rho, rho)
        E, dE, d2E = ergodic_E_Ce(g201, rho, derivs=True)
        E_ref = ergodic_E_Ce(ref, rho)
        phi, n_eval, off, ng = optimize_proposed_hybrid(A, rho, g201, E, dE, d2E)
        v_h = np.maximum(0, np.log2(1 + A * (1 - phi)) - np.interp(phi, ref, E_ref))   # 참값 기준 평가
        v_ref = np.empty(num_samples); v_201 = np.empty(num_samples)
        E201 = np.interp(g201, ref, E_ref)
        for s0 in range(0, num_samples, 1000):                     # 메모리 절약용 청크
            sl = slice(s0, s0 + 1000)
            v_ref[sl] = np.maximum(0, np.log2(1 + A[sl, None] * (1 - ref)[None, :]) - E_ref[None, :]).max(1)
            v_201[sl] = np.maximum(0, np.log2(1 + A[sl, None] * (1 - g201)[None, :]) - E201[None, :]).max(1)
        loss = v_ref - v_h
        print(f"{snr_db:>7} | {off.mean():>7.3f} | {ng.mean():>7.4f} | {loss.mean():>10.1e} | {loss.max():>10.1e} | "
              f"{(v_ref - v_201).max():>13.1e} | {n_eval.mean():>12.1f} | {int(n_eval.max()):>4}")


def sop_closed_form(A, phi, rho_e, Rs=RS):
    # 5.6절 조건부 SOP 닫힌형: P(Cb - Ce < Rs | h), 블록별 [N]
    #   theta = (1 + A(1-phi)) / 2^Rs - 1
    #   theta > 0 : (1-phi) / (theta*phi + 1 - phi) * exp(-theta / (rho_e (1-phi)))
    #   theta <= 0: 1 (Bob의 용량만으로도 Rs에 못 미침)
    phi = np.broadcast_to(np.asarray(phi, dtype=float), np.shape(A))
    theta = (1 + A * (1 - phi)) / 2 ** Rs - 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = (1 - phi) / (theta * phi + 1 - phi) * np.exp(-theta / (rho_e * (1 - phi)))
    return np.where(theta > 0, p, 1.0)


def verify_analysis(snr_db_list=(-10, 0, 10, 20, 30), phi_list=(0.0, 0.25, 0.5, 0.75, 0.95),
                    num_samples=200000, seed=3):
    # 해석식 검증: 같은 양을 식과 Monte Carlo로 각각 구해 비교
    #   A. 5.3절 E_g[Ce](phi) 준닫힌형 vs MC (미리 정한 phi, 탐색 구간 전체)
    #   B. 5.6절 조건부 SOP 닫힌형 vs MC (미리 정한 phi, ||h||^2)
    #   C. 고른 phi*에서 Proposed의 예측 성능(반해석) vs 실제 성능(MC)
    # MC는 X, Y를 직접 뽑지 않고 실제 채널과 beamforming으로 만든 B, D를 사용 -> 5.2절 분포 유도까지 함께 검증
    # z: |식 - MC| / MC 표준오차 (대략 3 이하면 통계적 잡음 범위)
    rng = np.random.default_rng(seed)
    phi_arr = np.array(phi_list, dtype=float)
    g201 = np.linspace(0.0, PHI_MAX, NUM_PHI)

    print(f"[해석식 검증 A] 5.3절 E_g[Ce](phi): 준닫힌형 vs MC (표본 {num_samples})")
    print(f"{'SNR(dB)':>7} | {'phi':>5} | {'식':>9} | {'MC':>9} | {'상대오차':>9} | {'z':>5}")
    worst_a = 0.0
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        _, B, D = generate_gains(rng, num_samples, rho, rho)
        E_formula = ergodic_E_Ce(phi_arr, rho)
        for phi, ef in zip(phi_arr, E_formula):
            ce = np.log2(1 + B * (1 - phi) / (1 + D * phi))
            mc, se = ce.mean(), ce.std() / np.sqrt(num_samples)
            z = abs(ef - mc) / se if se > 0 else 0.0
            worst_a = max(worst_a, z)
            rel = abs(ef - mc) / mc if mc > 0 else 0.0
            print(f"{snr_db:>7} | {phi:>5.2f} | {ef:>9.5f} | {mc:>9.5f} | {rel:>9.2e} | {z:>5.2f}")
    print(f"  -> 최대 z = {worst_a:.2f}")

    print(f"\n[해석식 검증 B] 5.6절 조건부 SOP: 닫힌형 vs MC (Rs = {RS})")
    print(f"{'SNR(dB)':>7} | {'phi':>5} | {'||h||^2':>7} | {'식':>9} | {'MC':>9} | {'z':>5}")
    worst_b = 0.0
    for snr_db in (10, 20, 30):                                          # 0 dB 이하는 대부분 theta <= 0이라 둘 다 1
        rho = 10 ** (snr_db / 10)
        _, B, D = generate_gains(rng, num_samples, rho, rho)
        for phi in (0.0, 0.5, 0.9):
            for x in (0.5, 2.0):
                A = rho * x                                                  # ||h||^2 = x로 고정
                p_formula = sop_closed_form(np.array([A]), phi, rho)[0]
                d = np.log2(1 + A * (1 - phi)) - np.log2(1 + B * (1 - phi) / (1 + D * phi))
                mc = np.mean(d < RS)
                se = np.sqrt(max(mc * (1 - mc), 1e-12) / num_samples)
                z = abs(p_formula - mc) / se
                worst_b = max(worst_b, z)
                print(f"{snr_db:>7} | {phi:>5.2f} | {x:>7.1f} | {p_formula:>9.5f} | {mc:>9.5f} | {z:>5.2f}")
    print(f"  -> 최대 z = {worst_b:.2f}")

    print(f"\n[해석식 검증 C] 고른 phi*에서 Proposed의 예측(반해석) vs 실제(MC)")
    print(f"{'SNR(dB)':>7} | {'전송률 예측':>10} | {'전송률 MC':>9} | {'z':>5} | {'SOP 예측':>9} | {'SOP MC':>9} | {'z':>5}")
    for snr_db in snr_db_list:
        rho = 10 ** (snr_db / 10)
        A, B, D = generate_gains(rng, num_samples, rho, rho)
        E, dE, d2E = ergodic_E_Ce(g201, rho, derivs=True)
        phi, _, _, _ = optimize_proposed_hybrid(A, rho, g201, E, dE, d2E)
        c_bar = np.log2(1 + A * (1 - phi)) - np.interp(phi, g201, E)          # 목적함수 값 (h만으로 계산)
        d = f_ch4(phi, A, B, D)                                              # 실제 g 위의 Cb - Ce
        r_mc_blocks = np.where(c_bar > 0, d, 0.0)
        r_pred, r_mc = np.maximum(0.0, c_bar).mean(), r_mc_blocks.mean()
        z_r = abs(r_pred - r_mc) / (r_mc_blocks.std() / np.sqrt(num_samples))
        s_pred, s_mc = sop_closed_form(A, phi, rho).mean(), np.mean(d < RS)
        se_s = np.sqrt(max(s_mc * (1 - s_mc), 1e-12) / num_samples)
        z_s = abs(s_pred - s_mc) / se_s
        print(f"{snr_db:>7} | {r_pred:>10.5f} | {r_mc:>9.5f} | {z_r:>5.2f} | {s_pred:>9.5f} | {s_mc:>9.5f} | {z_s:>5.2f}")


def optimize_static_phi(rho_b, phi_grid, E_Ce, n=STATIC_TRAIN_N, seed=STATIC_SEED, chunk=20000):
    # SNR-static: h 순시값 없이 채널 통계만으로 SNR당 하나의 phi 결정
    #   phi_static = argmax_phi E_h[ [Cb(phi) - E_g[Ce](phi)]^+ ]  (달성 가능 전송률의 기댓값)
    # ||h||^2 ~ Gamma(Nt, 1) (h ~ CN(0, I), Nt=2) 독립 표본으로 E_h를 추정 -> 평가 표본과 분리해 편향 방지
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(phi_grid))
    for s0 in range(0, n, chunk):
        A = rho_b * rng.gamma(NT, 1.0, size=min(chunk, n - s0))
        acc += np.maximum(0.0, np.log2(1 + A[:, None] * (1 - phi_grid)[None, :]) - E_Ce[None, :]).sum(axis=0)
    return phi_grid[np.argmax(acc)]


def run_simulation(num_samples=NUM_SAMPLES, Rs=RS, sigma_b2=SIGMA_B2, sigma_e2=SIGMA_E2, snr_db_range=SNR_DB_RANGE, num_phi=NUM_PHI, seed=SEED):
    rng = np.random.default_rng(seed)               # 시드 고정 난수 생성기
    phi_grid = np.linspace(0.0, PHI_MAX, num_phi)   # phi 후보값 배열 [G]

    # ----- 결과 누적용 dictionary -----
    res = {k: [] for k in ["snr", "cs_fixed", "cs_genie", "cs_proposed", "sop_fixed", "sop_genie", "sop_proposed", "phi_genie", "phi_proposed", "evals_genie", "safe_frac", "evals_proposed", "an_off_frac", "cs_static", "sop_static", "phi_static"]}

    # ----- 진행 상황 출력 -----
    print(f"시뮬레이션 시작 (Samples: {num_samples}, Rs: {Rs} bps/Hz)")
    print("-" * 97)
    print(f"{'SNR(dB)':>7} | {'Cs_fixed':>12} | {'Cs_genie':>12} | {'Cs_proposed':>12} | " f"{'SOP_fixed':>12} | {'SOP_genie':>12} | {'SOP_proposed':>12}")
    print("-" * 97)

    # ----- 수식 계산 -----
    for snr_db in snr_db_range:
        # 송신 SNR은 Bob 잡음 분산 기준: rho_b = P_total / sigma_b^2
        P_total = (10 ** (snr_db / 10)) * sigma_b2  # dB -> 선형 총 송신 전력
        rho_b = P_total / sigma_b2                  # Bob 기준 송신 SNR
        rho_e = P_total / sigma_e2                  # Eve 기준 송신 SNR (등분산 설정 시 rho_b와 동일)

        A, B, D = generate_gains(rng, num_samples, rho_b, rho_e)   # 식 (3.8)

        # 5장 목적함수의 E_g[Ce]: h 무관 -> SNR당 1회 계산해 Fixed, Proposed가 공유
        E_Ce, dE_Ce, d2E_Ce = ergodic_E_Ce(phi_grid, rho_e, derivs=True)  # [G]×3

        # ===== 1. Fixed: 정보 신호와 인공 잡음 신호의 전력비 고정 (Eve CSI 없음) =====
        d_fixed = f_ch4(FIXED_PHI, A, B, D)                                   # 블록별 Cb - Ce
        tx_fixed = np.log2(1 + A * (1 - FIXED_PHI)) - np.interp(FIXED_PHI, phi_grid, E_Ce) > 0  # h만으로 전송 여부
        cs_fixed = np.where(tx_fixed, d_fixed, 0.0)                           # 달성 가능 (음수 블록 포함)

        # ===== 1-2. SNR-static: SNR마다 채널 통계로 정한 고정 phi (Eve CSI 없음, h 적응 없음) =====
        phi_static = optimize_static_phi(rho_b, phi_grid, E_Ce)
        d_static = f_ch4(phi_static, A, B, D)
        tx_static = np.log2(1 + A * (1 - phi_static)) - np.interp(phi_static, phi_grid, E_Ce) > 0
        cs_static = np.where(tx_static, d_static, 0.0)                        # 달성 가능 (Fixed와 같은 규칙)

        # ===== 2. Genie-aided: Eve의 CSI를 알고 4장 조건부 하이브리드로 phi 선택 =====
        phi_genie, n_eval_genie, safe = optimize_genie_hybrid(A, B, D)
        d_genie = f_ch4(phi_genie, A, B, D)
        cs_genie = np.maximum(0.0, d_genie)                                   # Eve CSI로 블록별 판정 가능

        # ===== 3. Proposed: Eve CSI(g) 미사용, 5장 목적함수 최대화로 phi 선택 =====
        # 목적함수 C_bar(phi) = Cb(phi|h) - E_g[Ce](phi), phi*=0 판정(닫힌형) + 뉴턴법 (5.4, 5.5절)
        phi_proposed, n_eval_prop, an_off, _ = optimize_proposed_hybrid(A, rho_e, phi_grid, E_Ce, dE_Ce, d2E_Ce)
        d_prop = f_ch4(phi_proposed, A, B, D)                                 # 실제 g 위에서의 블록별 Cb - Ce
        c_bar_prop = np.log2(1 + A * (1 - phi_proposed)) - np.interp(phi_proposed, phi_grid, E_Ce)
        tx_prop = c_bar_prop > 0                                              # h만으로 전송 여부
        cs_proposed = np.where(tx_prop, d_prop, 0.0)                          # 달성 가능 (음수 블록 포함)

        # ----- SNR별 지표 집계 -----
        res["snr"].append(snr_db)
        res["cs_fixed"].append(cs_fixed.mean())          # 에르고딕 보안 용량 (Fixed, Proposed: 달성 가능 전송률)
        res["cs_genie"].append(cs_genie.mean())
        res["cs_static"].append(cs_static.mean())
        res["cs_proposed"].append(cs_proposed.mean())
        res["sop_fixed"].append(np.mean(d_fixed < Rs))   # SOP = P(Cb - Ce < Rs)
        res["sop_genie"].append(np.mean(d_genie < Rs))
        res["sop_static"].append(np.mean(d_static < Rs))
        res["sop_proposed"].append(np.mean(d_prop < Rs))
        res["phi_genie"].append(phi_genie.mean())        # 평균 선택 phi
        res["phi_proposed"].append(phi_proposed.mean())
        res["phi_static"].append(phi_static)
        res["evals_genie"].append(n_eval_genie.mean())   # Genie 결정당 평균 평가 횟수
        res["safe_frac"].append(safe.mean())             # 안전 구간(D > B/2) 비율
        res["evals_proposed"].append(n_eval_prop.mean()) # Proposed 결정당 평균 평가 횟수
        res["an_off_frac"].append(an_off.mean())         # 인공 잡음 미사용(phi*=0) 판정 비율

        # ----- 현재 SNR 행 콘솔 출력 -----
        print(
            f"{snr_db:>7} | {res['cs_fixed'][-1]:>12.4f} | {res['cs_genie'][-1]:>12.4f} | "
            f"{res['cs_proposed'][-1]:>12.4f} | {res['sop_fixed'][-1]:>12.4f} | "
            f"{res['sop_genie'][-1]:>12.4f} | {res['sop_proposed'][-1]:>12.4f}"
        )

    return res


def plot_results(res, Rs=RS, num_samples=NUM_SAMPLES):
    snr = np.array(res["snr"])
    floor = max(1.0 / num_samples, 1e-6)  # semilogy에서 0 방지용 바닥값

    # 색상·스타일 상수 (기법별 통일)
    C_FIX, C_GEN, C_PRO, C_STA = "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"

    plt.figure(figsize=(13, 10))

    # ===== (1) 송신 SNR에 따른 평균 보안 용량 =====
    plt.subplot(2, 2, 1)
    plt.plot(snr, res["cs_fixed"], "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.6, markersize=6, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.plot(snr, res["cs_genie"], "^-.", color=C_GEN, label="Genie-aided (Full Eve CSI)", linewidth=1.8, markersize=7, zorder=2)
    plt.plot(snr, res["cs_static"], "x:", color=C_STA, label="SNR-static", linewidth=1.6, markersize=7, zorder=4)
    plt.plot(snr, res["cs_proposed"], "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.xlabel("Transmit SNR (dB)")
    plt.ylabel("Ergodic Secrecy Rate (bps/Hz)")
    plt.title("Ergodic Secrecy Rate")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (2) 송신 SNR에 따른 전력 분배 비율 phi =====
    plt.subplot(2, 2, 2)
    plt.axhline(FIXED_PHI, color=C_FIX, linestyle="-", linewidth=1.6, label="Fixed (phi=0.5)", zorder=1)
    plt.plot(snr, res["phi_genie"], "v--", color=C_GEN, label="Genie-aided", linewidth=1.5, zorder=2)
    plt.plot(snr, res["phi_static"], "x:", color=C_STA, label="SNR-static", linewidth=1.6, markersize=7, zorder=4)
    plt.plot(snr, res["phi_proposed"], "D-", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.xlabel("Transmit SNR (dB)")
    plt.ylabel("Average Optimal Power Ratio (phi*)")
    plt.title("AN Power Allocation Strategy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (3) 상한선 대비 보안 용량 격차 =====
    gap_proposed = np.array(res["cs_genie"]) - np.array(res["cs_proposed"])
    gap_fixed = np.array(res["cs_genie"]) - np.array(res["cs_fixed"])
    gap_static = np.array(res["cs_genie"]) - np.array(res["cs_static"])
    plt.subplot(2, 2, 3)
    plt.plot(snr, gap_fixed, "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.8, markersize=6, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.plot(snr, gap_static, "x:", color=C_STA, label="SNR-static", linewidth=1.6, markersize=7, zorder=4)
    plt.plot(snr, gap_proposed, "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.xlabel("Transmit SNR (dB)")
    plt.ylabel("Rate Gap from Genie (bps/Hz)")
    plt.title("Gap to Upper Bound (lower = closer)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (4) 송신 SNR에 따른 보안 중단 확률(Log Scale) =====
    plt.subplot(2, 2, 4)
    plt.semilogy(snr, np.maximum(res["sop_fixed"], floor), "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.8, markersize=6, markerfacecolor="none", markeredgewidth=1.5, zorder=2)
    plt.semilogy(snr, np.maximum(res["sop_genie"], floor), "^-.", color=C_GEN, label="Genie-aided (Full Eve CSI)", linewidth=1.8, markersize=7, zorder=2)
    plt.semilogy(snr, np.maximum(res["sop_static"], floor), "x:", color=C_STA, label="SNR-static", linewidth=1.6, markersize=7, zorder=4)
    plt.semilogy(snr, np.maximum(res["sop_proposed"], floor), "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=1.6, markersize=6, markerfacecolor="none", markeredgewidth=1.5, zorder=3)
    plt.xlabel("Transmit SNR (dB)")
    plt.ylabel("SOP (Log Scale)")
    plt.title(f"Secrecy Outage Probability (Rs={Rs})")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 검증
    if RUN_VERIFY:
        verify_ch4_hybrid()
        verify_ch5_hybrid()
        verify_analysis()

    # 시뮬레이션 수행
    plot_results(run_simulation())
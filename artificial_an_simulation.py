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
# ====================================


# ----- HyperParameters --------------
NUM_SAMPLES = 50000                     # Monte-Carlo 채널 표본 수
RS = 1.0                                # 목표 보안 전송률 (bps/Hz)
SIGMA_N2 = 1e-10                        # 수신단 열잡음 분산
NUM_PHI = 201                           # Grid search 해상도(정밀도)
SEED = 0                                # 난수 시드 (재현성)
SNR_DB_RANGE = np.arange(-20, 31, 5)    # 송신 SNR 구간 (dB)

NT = 2              # 송신 안테나 수 (현 공식은 2로 고정된 경우에 한정됨, TODO: 이후 N_t > 2인 경우로 확장 예정)

FIXED_PHI = 0.5     # 전력비 고정 기법의 전력비
PHI_MAX = 0.99      # phi 탐색 상한 (1.0인 경우 신호 전력이 0이므로 제외)

EG_QUAD_N = 400     # E_g[Ce]의 Y(AN 성분) 수치적분 격자점 수
EG_QUAD_YMAX = 60.0 # Y 적분 상한 (Exp(1) tail, e^{-60}은 무시 가능)

NEWTON_MAX_ITER = 20    # 뉴턴법 최대 반복 수
NEWTON_TOL = 1e-7       # 뉴턴법 step 크기
PHI_MAX_NEWTON = 0.9    # 뉴턴 적용 영역 상한 (해당 범위 외 극단은 grid 사용)
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


def ergodic_E_Ce(phi_grid, rho, derivs=False):
    # E_g[Ce](phi) 를 phi_grid 전체에 대해 계산 (bps/Hz)
    # Nt = 2에서 Eve의 신호 / AN 이득이 지수분포를 따름을 이용해 신호 성분은 지수적분(E1)으로 해석적으로 처리하고 AN 성분만 수치적분
    # h와 무관하므로 SNR당 1회만 계산
    # derivs=True 이면 phi에 대한 1·2계 해석적 도함수도 함께 반환 (뉴턴법용)
    # 고 SNR에서 피적분함수가 y=0 근처에서 급변하므로 원점 근처를 촘촘히 배치 (비균일 격자)
    y = np.concatenate([[0.0], np.geomspace(1e-6, EG_QUAD_YMAX, EG_QUAD_N - 1)])[None, :]  # [1, Qy]
    wy = np.exp(-y)                                             # Exp(1) 가중치
    one_minus = (1.0 - phi_grid)[:, None]                       # [G, 1]
    c = rho * one_minus / (1.0 + rho * phi_grid[:, None] * y)   # [G, Qy]
    c = np.maximum(c, 1e-15)
    Fu = _stable_exp_e1(1.0 / c)                               # F(u)=e^u E1(u), u=1/c [G,Qy]
    E_nat = np.trapezoid(Fu * wy, y[0], axis=1)                # [G]
    E_Ce = E_nat / np.log(2.0)                                 # nat -> bps/Hz
    if not derivs:
        return E_Ce

    # phi 도함수
    u = 1.0 / c                                         # [G,Qy]
    up = (1.0 + rho * y) / (rho * one_minus**2)         # u'(phi)  [G,Qy]
    upp = 2.0 * (1.0 + rho * y) / (rho * one_minus**3)  # u''(phi) [G,Qy]
    Fp = Fu - 1.0 / u                                   # F'(u)
    Fpp = Fu - 1.0 / u + 1.0 / u**2                     # F''(u)
    dE = np.trapezoid((Fp * up) * wy, y[0], axis=1) / np.log(2.0)                   # E_Ce'(phi) [G]
    d2E = np.trapezoid((Fpp * up**2 + Fp * upp) * wy, y[0], axis=1) / np.log(2.0)   # E_Ce''(phi) [G]
    return E_Ce, dE, d2E


def optimize_proposed_hybrid(Kb, rho, phi_grid, E_Ce, dE_Ce, d2E_Ce):
    # 조건부 Ergodic Cs를 최대화하는 phi를 채널별로 탐색
    # 하이브리드 방식: 뉴턴법을 적용하되, 극단은 grid로 보호
    # E_Ce 계열은 phi_grid 위 사전계산값 -> 뉴턴 중 격자 선형보간으로 연속 평가
    ln2 = np.log(2.0); N = len(Kb)

    # Grid: 안전장치 및 뉴턴법 초기값
    one_minus = (1.0 - phi_grid)[None, :]
    Cb_grid = np.log(1.0 + Kb[:, None] * one_minus) / ln2       # [N,G]
    cs_grid = np.maximum(0.0, Cb_grid - E_Ce[None, :])          # [N,G]
    idx_grid = np.argmax(cs_grid, axis=1)
    phi_grid_star = phi_grid[idx_grid]                          # grid 해 [N]

    # 뉴턴법: grid 해에서 출발해 관심영역 내에서 정밀화, 벡터화 적용
    phi = phi_grid_star.copy()
    for _ in range(NEWTON_MAX_ITER):
        within = (phi > 1e-4) & (phi < PHI_MAX_NEWTON)         # 관심영역만 갱신
        d = 1.0 + Kb * (1.0 - phi)
        Cbp = (-Kb / d) / ln2
        Cbpp = (-Kb**2 / d**2) / ln2
        csp = Cbp - np.interp(phi, phi_grid, dE_Ce)            # Cs_bar'
        cspp = Cbpp - np.interp(phi, phi_grid, d2E_Ce)         # Cs_bar''
        step = np.where(np.abs(cspp) > 1e-12, csp / cspp, 0.0)
        phi_new = np.clip(phi - step, 1e-4, PHI_MAX_NEWTON)
        phi = np.where(within, phi_new, phi)
        if np.max(np.abs(np.where(within, step, 0.0))) < NEWTON_TOL:
            break

    # 뉴턴 해 평가 후 grid와 비교해 더 나은 쪽 채택
    d = 1.0 + Kb * (1.0 - phi)
    cs_newton = np.maximum(0.0, np.log(d) / ln2 - np.interp(phi, phi_grid, E_Ce))
    cs_grid_best = cs_grid[np.arange(N), idx_grid]
    phi_star = np.where(cs_newton >= cs_grid_best, phi, phi_grid_star)
    return phi_star


def run_simulation(num_samples=NUM_SAMPLES, Rs=RS, sigma_n2=SIGMA_N2, snr_db_range=SNR_DB_RANGE, num_phi=NUM_PHI, seed=SEED):
    rng = np.random.default_rng(seed)               # 시드 고정 난수 생성기
    phi_grid = np.linspace(0.0, PHI_MAX, num_phi)   # phi 후보값 배열 [G]

    # ----- 결과 누적용 dictionary -----
    res = {k: [] for k in ["snr", "cs_fixed", "cs_genie", "cs_proposed", "sop_fixed", "sop_genie", "sop_proposed", "phi_genie", "phi_proposed"]}

    # ----- 진행 상황 출력 -----
    print(f"시뮬레이션 시작 (Samples: {num_samples}, Rs: {Rs} bps/Hz)")
    print("-" * 97)
    print(f"{'SNR(dB)':>7} | {'Cs_fixed':>12} | {'Cs_genie':>12} | {'Cs_proposed':>12} | " f"{'SOP_fixed':>12} | {'SOP_genie':>12} | {'SOP_proposed':>12}")
    print("-" * 97)

    # ----- 수식 계산 -----
    for snr_db in snr_db_range:
        P_total = (10 ** (snr_db / 10)) * sigma_n2  # dB -> 선형 총 송신 전력 변환
        rho = P_total / sigma_n2                    # 송신 SNR

        # ----- h, g ~ CN(0, I) 독립 Rayleigh 채널. 배열 [N, Nt]의 각 행이 Nt x 1 열벡터 하나 -----
        h = (rng.standard_normal((num_samples, NT)) + 1j * rng.standard_normal((num_samples, NT))) / np.sqrt(2)  # Bob 채널
        g = (rng.standard_normal((num_samples, NT)) + 1j * rng.standard_normal((num_samples, NT))) / np.sqrt(2)  # Eve 채널

        norm_h = np.linalg.norm(h, axis=1, keepdims=True)   # 채널별 ||h|| [N,1]
        w_s = h / norm_h                                    # MRT: w_s = h / ||h||

        # 영공간 빔포밍 방향: h^H w_z = 0, w_s^H w_z = 0
        # (단, 해당 공식은 N_t=2에서만 성립 -> TODO: 이후 N_t > 2인 경우로 확장 예정)
        w_z = np.stack([-np.conj(h[:, 1]), np.conj(h[:, 0])], axis=1) / norm_h
        _check_beamformers(h, w_s, w_z)

        # ----- 유효 채널 이득 계수 -----
        hw_s = hdot(h, w_s)  # h^H w_s (= ||h||)
        gw_s = hdot(g, w_s)  # g^H w_s : Eve의 수신 신호 성분
        gw_z = hdot(g, w_z)  # g^H w_z : Eve의 수신 AN 성분

        Kb = rho * np.abs(hw_s) ** 2    # Bob 유효 SNR 계수 [N]
        Kes = rho * np.abs(gw_s) ** 2   # Eve 신호 이득 [N]
        Kez = rho * np.abs(gw_z) ** 2   # Eve AN 방해 이득 [N]

        one_minus = (1.0 - phi_grid)[None, :]  # (1 - phi) 브로드캐스트용 [1, G]

        # ----- 채널 × phi 조합별 실현 보안 용량 Cs [N, G] -----
        Cb_mat = np.log2(1 + Kb[:, None] * one_minus)  # Bob 용량
        sinr_e = (Kes[:, None] * one_minus) / (1 + Kez[:, None] * phi_grid[None, :])
        Ce_mat = np.log2(1 + sinr_e)  # Eve 용량
        Cs_mat = np.maximum(0.0, Cb_mat - Ce_mat)  # 보안 용량 (음수는 0으로 clip)

        # ===== 1. Fixed: 정보 신호와 인공 잡음 신호의 전력비 고정 =====
        idx_fixed = int(np.argmin(np.abs(phi_grid - FIXED_PHI)))  # 격자상 최근접 위치
        cs_fixed = Cs_mat[:, idx_fixed]

        # ===== 2. Genie-aided: Eve의 CSI g를 알고 있다고 가정 =====
        idx_genie = np.argmax(Cs_mat, axis=1)                   # 채널별 최적 phi 위치
        cs_genie = Cs_mat[np.arange(num_samples), idx_genie]    # 해당 위치의 Cs 값
        phi_genie = phi_grid[idx_genie]                         # 해당 위치의 phi 값

        # ===== (3) Proposed: Eve CSI(g) 미사용, 조건부 Ergodic Cs 최대화로 phi 선택 =====
        # h는 알고(순시), g만 분포로 처리 -> 목적함수: E_g[Cs(phi) | h]
        #   Cb(phi|h) = log2(1 + Kb(1-phi))  : h로 직접 (g 무관)
        #   E_g[Ce](phi)                     : g 분포로 평균 (h 무관 -> SNR당 1회)
        # 오목성 분석에 근거한 하이브리드(기본 뉴턴 + 극단 grid)로 최적화.
        E_Ce, dE_Ce, d2E_Ce = ergodic_E_Ce(phi_grid, rho, derivs=True)  # [G]×3, h 무관하여 공유
        phi_proposed = optimize_proposed_hybrid(Kb, rho, phi_grid, E_Ce, dE_Ce, d2E_Ce)  # [N]
        # 실현 성능: blind 선택한 phi를 실제 g 위에서 평가 (phi는 격자 밖일 수 있어 직접 계산)
        sinr_e_prop = (Kes * (1 - phi_proposed)) / (1 + Kez * phi_proposed)
        cs_proposed = np.maximum(0.0, np.log2(1 + Kb * (1 - phi_proposed)) - np.log2(1 + sinr_e_prop))

        # ----- SNR별 지표 집계 -----
        res["snr"].append(snr_db)
        res["cs_fixed"].append(cs_fixed.mean())  # 평균(ergodic) 보안 용량
        res["cs_genie"].append(cs_genie.mean())
        res["cs_proposed"].append(cs_proposed.mean())
        res["sop_fixed"].append(np.mean(cs_fixed < Rs))  # SOP = P(Cs < Rs)
        res["sop_genie"].append(np.mean(cs_genie < Rs))
        res["sop_proposed"].append(np.mean(cs_proposed < Rs))
        res["phi_genie"].append(phi_genie.mean())  # 평균 선택 phi
        res["phi_proposed"].append(phi_proposed.mean())

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
    C_FIX, C_GEN, C_PRO = "#f59e0b", "#10b981", "#3b82f6"

    plt.figure(figsize=(13, 10))

    # ===== (1) 송신 SNR에 따른 평균 보안 용량 =====
    plt.subplot(2, 2, 1)
    plt.plot(snr, res["cs_fixed"], "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.6, markersize=6, markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    plt.plot(snr, res["cs_genie"], "^-", color=C_GEN, label="Genie-aided (Full Eve CSI)", linewidth=4.5, markersize=11, alpha=0.9, zorder=1)
    plt.plot(snr, res["cs_proposed"], "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2, zorder=2)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("Ergodic Secrecy Capacity (bps/Hz)")
    plt.title("Ergodic Secrecy Capacity")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (2) 송신 SNR에 따른 전력 분배 비율 phi =====
    plt.subplot(2, 2, 2)
    plt.axhline(FIXED_PHI, color=C_FIX, linestyle="-", linewidth=1.6, label="Fixed (phi=0.5)", zorder=1)
    plt.plot(snr, res["phi_genie"], "v--", color=C_GEN, label="Genie-aided", linewidth=1.5, zorder=2)
    plt.plot(snr, res["phi_proposed"], "D-", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2, zorder=3)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("Average Optimal Power Ratio (phi*)")
    plt.title("AN Power Allocation Strategy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (3) 상한선 대비 보안 용량 격차 =====
    gap_proposed = np.array(res["cs_genie"]) - np.array(res["cs_proposed"])
    gap_fixed = np.array(res["cs_genie"]) - np.array(res["cs_fixed"])
    plt.subplot(2, 2, 3)
    plt.plot(snr, gap_fixed, "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.8, markersize=6, markerfacecolor="white", markeredgewidth=1.5)
    plt.plot(snr, gap_proposed, "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=2)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("Capacity Gap from Genie (bps/Hz)")
    plt.title("Gap to Upper Bound (lower = closer)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ===== (4) 송신 SNR에 따른 보안 중단 확률(Log Scale) =====
    plt.subplot(2, 2, 4)
    plt.semilogy(snr, np.maximum(res["sop_fixed"], floor), "o-", color=C_FIX, label="Fixed (phi=0.5)", linewidth=1.8, zorder=2)
    plt.semilogy(snr, np.maximum(res["sop_genie"], floor), "^-", color=C_GEN, label="Genie-aided (Full Eve CSI)", linewidth=4.5, markersize=11, alpha=0.9, zorder=1)
    plt.semilogy(snr, np.maximum(res["sop_proposed"], floor), "s--", color=C_PRO, label="Proposed (No Eve CSI)", linewidth=1.6, markersize=6, markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("SOP (Log Scale)")
    plt.title(f"Secrecy Outage Probability (Rs={Rs})")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_simulation()
    plot_results(results)
import numpy as np
import matplotlib.pyplot as plt

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
PHI_MAX = 0.99      # phi 탐색 상한 (1.0의 경우 신호 전력이 0이므로 제외)
# ------------------------------------


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

        # ----- 각 원소가 CN(0,1)인 Rayleigh 페이딩 채널 [N, Nt] 생성 -----
        h = (rng.standard_normal((num_samples, NT)) + 1j * rng.standard_normal((num_samples, NT))) / np.sqrt(2)  # Bob 채널
        g = (rng.standard_normal((num_samples, NT)) + 1j * rng.standard_normal((num_samples, NT))) / np.sqrt(2)  # Eve 채널

        norm_h = np.linalg.norm(h, axis=1, keepdims=True)   # 채널별 ||h|| [N,1]
        w_s = np.conj(h) / norm_h                           # MRT 빔포밍 방향

        # 영공간 빔포밍 방향 (w_z ⊥ h)
        # (단, 해당 공식은 N_t=2에서만 성립 -> TODO: 이후 N_t > 2인 경우로 확장 예정)
        w_z = np.stack([-np.conj(h[:, 1]), np.conj(h[:, 0])], axis=1) / norm_h

        # ----- 유효 채널 이득 계수 -----
        hw_s = np.sum(h * w_s, axis=1)  # Bob의 수신 신호 성분 (= ||h||)
        gw_s = np.sum(g * w_s, axis=1)  # Eve의 수신 신호 성분
        gw_z = np.sum(g * w_z, axis=1)  # Eve의 수신 AN 성분

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

        # ===== (3) Proposed: g 미사용, 조건부 SOP 최소화로 phi 선택 (제안 기법) =====
        A = Kb[:, None]                             # Bob 유효 SNR 계수 [N,1]
        gth = (1 + A * one_minus) / (2.0**Rs) - 1   # Eve 요구 SINR 임계값 [N,G]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a_term = gth / (rho * one_minus)                # SOP 지수 항
            b_term = gth * phi_grid[None, :] / one_minus    # SOP 분모 항
            sop_grid = np.exp(-a_term) / (1 + b_term)       # 조건부 SOP [N,G]
        sop_grid = np.where(gth <= 0, 1.0, sop_grid)    # Bob이 Rs 미달 시 중단
        idx_proposed = np.argmin(sop_grid, axis=1)      # SOP 최소화 phi 위치
        phi_proposed = phi_grid[idx_proposed]
        cs_proposed = Cs_mat[np.arange(num_samples), idx_proposed]  # 실현 성능: blind 선택 phi를 실제 g 위에서 평가

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
    snr = res["snr"]
    floor = max(1.0 / num_samples, 1e-6)  # semilogy에서 0 방지용 바닥값

    plt.figure(figsize=(18, 5))

    # ----- 송신 SNR에 따른 평균 보안 용량 -----
    plt.subplot(1, 3, 1)
    plt.plot(snr, res["cs_fixed"], "o-", color="#f59e0b", label="Fixed (phi=0.5)", linewidth=1.6, markersize=6, markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    plt.plot(snr, res["cs_genie"], "^-", color="#10b981", label="Genie-aided (Full Eve CSI)", linewidth=4.5, markersize=11, alpha=0.9, zorder=1)
    plt.plot(snr, res["cs_proposed"], "s--", color="#3b82f6", label="Proposed (No Eve CSI)", linewidth=2, zorder=2)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("Ergodic Secrecy Capacity (bps/Hz)")
    plt.title("Ergodic Secrecy Capacity")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ----- 송신 SNR에 따른 전력 분배 비율 phi -----
    plt.subplot(1, 3, 2)
    plt.plot(snr, res["phi_genie"], "v--", color="#10b981", label="Genie-aided", linewidth=1.5)
    plt.plot(snr, res["phi_proposed"], "D-", color="#3b82f6", label="Proposed (No Eve CSI)", linewidth=2)
    plt.xlabel("Average Transmit SNR (dB)")
    plt.ylabel("Average Optimal Power Ratio (phi*)")
    plt.title("AN Power Allocation Strategy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    # ----- 송신 SNR에 따른 보안 중단 확률(Log Scale) -----
    plt.subplot(1, 3, 3)
    plt.semilogy(snr, np.maximum(res["sop_fixed"], floor), "o-", color="#f59e0b", label="Fixed (phi=0.5)", linewidth=1.8, zorder=2)
    plt.semilogy(snr, np.maximum(res["sop_genie"], floor), "^-", color="#10b981", label="Genie-aided (Full Eve CSI)", linewidth=4.5, markersize=11, alpha=0.9, zorder=1)
    plt.semilogy(
        snr, np.maximum(res["sop_proposed"], floor), "s--", color="#1e3a8a", label="Proposed (No Eve CSI)", linewidth=1.6, markersize=6, markerfacecolor="white", markeredgewidth=1.5, zorder=3
    )
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

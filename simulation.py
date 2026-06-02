import numpy as np
import matplotlib.pyplot as plt

# === 조건부 하이브리드 기반 적응형 최적화 함수 ===
def solve_optimal_phi_hybrid(P_total, sigma_n2, h, g, tol=1e-6, max_iter=15):
    norm_h = np.linalg.norm(h)
    w_s = h.conj().T / norm_h
    w_z = np.array([-h[1].conj(), h[0].conj()])
    w_z = w_z / np.linalg.norm(w_z)

    # 파라미터 계산 (B: Kes, D: Kez)
    Kb = (P_total * np.abs(np.dot(h, w_s))**2) / sigma_n2
    Kes = (P_total * np.abs(np.dot(g, w_s))**2) / sigma_n2
    Kez = (P_total * np.abs(np.dot(g, w_z))**2) / sigma_n2

    # -----------------------------------------------------------
    # [Case 1] D > B/2: 전 구간 오목성 보장 -> 뉴턴법 직행 (Fast-track)
    # -----------------------------------------------------------
    if Kez > (Kes / 2.0):
        phi_n = 0.5
        for _ in range(max_iter):
            den_b = 1 + Kb * (1 - phi_n)
            den_e1 = 1 + Kes + (Kez - Kes) * phi_n
            den_e2 = 1 + Kez * phi_n
            
            f_prime = (-Kb / den_b) - ((Kez - Kes) / den_e1) + (Kez / den_e2)
            f_double_prime = (-Kb**2 / den_b**2) + ((Kez - Kes)**2 / den_e1**2) - (Kez**2 / den_e2**2)
            
            if f_double_prime >= 0: # 수학적으로 발생하지 않으나 안전장치
                break
                
            phi_next = phi_n - (f_prime / f_double_prime)
            phi_next = np.clip(phi_next, 0, 0.99)
            
            if np.abs(phi_next - phi_n) < tol:
                phi_n = phi_next
                break
            phi_n = phi_next
            
        den_b = 1 + Kb * (1 - phi_n)
        den_e1 = 1 + Kes + (Kez - Kes) * phi_n
        den_e2 = 1 + Kez * phi_n
        Cb = np.log2(den_b)
        Ce = np.log2(den_e1 / den_e2)
        return phi_n, max(0, Cb - Ce), Kb, Kes, Kez

    # -----------------------------------------------------------
    # [Case 2] D < B/2: 오목성 깨질 위험 존재 -> Grid Search + Newton 하이브리드
    # -----------------------------------------------------------
    else:
        num_grid_points = 20 # 연산 속도를 위해 20 포인트로 타협
        phi_grid = np.linspace(0, 0.99, num_grid_points)
        
        best_cs_grid = 0
        best_phi_grid = 0
        concave_candidates = []

        # 1. Grid Search를 통한 조건 교집합(Cs > 0 & f'' < 0) 탐색
        for phi in phi_grid:
            den_b = 1 + Kb * (1 - phi)
            den_e1 = 1 + Kes + (Kez - Kes) * phi
            den_e2 = 1 + Kez * phi
            
            Cb = np.log2(den_b)
            Ce = np.log2(den_e1 / den_e2)
            cs = max(0, Cb - Ce)
            
            if cs > best_cs_grid:
                best_cs_grid = cs
                best_phi_grid = phi
                
            if cs > 0:
                f_double_prime = (-Kb**2 / den_b**2) + ((Kez - Kes)**2 / den_e1**2) - (Kez**2 / den_e2**2)
                if f_double_prime < 0:
                    concave_candidates.append(phi)

        # 2. 오목성 보장 구간에서 뉴턴법 실행
        best_cs_newton = 0
        best_phi_newton = 0
        
        if concave_candidates:
            # Grid Search에서 성능이 가장 좋았던 값을 초기값으로 설정
            phi_n = best_phi_grid if best_phi_grid in concave_candidates else concave_candidates[len(concave_candidates)//2]
            
            for _ in range(max_iter):
                den_b = 1 + Kb * (1 - phi_n)
                den_e1 = 1 + Kes + (Kez - Kes) * phi_n
                den_e2 = 1 + Kez * phi_n
                
                f_prime = (-Kb / den_b) - ((Kez - Kes) / den_e1) + (Kez / den_e2)
                f_double_prime = (-Kb**2 / den_b**2) + ((Kez - Kes)**2 / den_e1**2) - (Kez**2 / den_e2**2)
                
                if f_double_prime >= 0: # 오목성 상실 시 즉시 중단
                    break
                    
                phi_next = phi_n - (f_prime / f_double_prime)
                phi_next = np.clip(phi_next, 0, 0.99)
                
                if np.abs(phi_next - phi_n) < tol:
                    phi_n = phi_next
                    break
                phi_n = phi_next
                
            den_b = 1 + Kb * (1 - phi_n)
            den_e1 = 1 + Kes + (Kez - Kes) * phi_n
            den_e2 = 1 + Kez * phi_n
            Cb = np.log2(den_b)
            Ce = np.log2(den_e1 / den_e2)
            if Cb - Ce > 0:
                best_cs_newton = Cb - Ce
                best_phi_newton = phi_n

        # 3. 최댓값 선택 (안전장치)
        if best_cs_newton > best_cs_grid:
            return best_phi_newton, best_cs_newton, Kb, Kes, Kez
        else:
            return best_phi_grid, best_cs_grid, Kb, Kes, Kez


# === 몬테카를로 시뮬레이션 및 SOP 계산 ===
def run_main_simulation():
    num_samples = 100000  # 사전 테스트를 위해 10,000으로 축소 (최종 시 100,000 복구)
    snr_db_range = np.arange(-20, 31, 5)
    sigma_n2 = 1e-10
    Nt = 2
    Rs = 1.0  # 목표 보안 전송률
    
    results_snr = []
    results_cs_adap = []
    results_cs_fixed = []
    results_sop_adap = []
    results_sop_fixed = []
    results_phi_avg = []

    print(f"시뮬레이션 시작 (Samples: {num_samples}, Rs: {Rs} bps/Hz)")
    print("-" * 85)
    print(f"{'SNR(dB)':>7} | {'Avg Cs(Adap)':>12} | {'Avg Cs(Fixed)':>12} | {'SOP(Adap)':>10} | {'SOP(Fixed)':>10}")
    print("-" * 85)

    for snr_db in snr_db_range:
        P_total = (10**(snr_db / 10)) * sigma_n2
        
        cs_list_adap = []
        cs_list_fixed = []
        outage_adap = 0
        outage_fixed = 0
        phi_star_list = []
        
        for _ in range(num_samples):
            h = (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)
            g = (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)
            
            # Hybrid Adaptive (Proposed) 함수 호출 변경
            phi_star, cs_adap, Kb, Kes, Kez = solve_optimal_phi_hybrid(P_total, sigma_n2, h, g)
            
            # Fixed: phi = 0.5 (Baseline)
            cs_fixed = max(0, np.log2(1 + Kb*(1-0.5)) - np.log2(1 + (Kes*(1-0.5))/(Kez*0.5 + 1)))
            
            if cs_adap < Rs: outage_adap += 1
            if cs_fixed < Rs: outage_fixed += 1
            
            cs_list_adap.append(cs_adap)
            cs_list_fixed.append(cs_fixed)
            phi_star_list.append(phi_star)
            
        results_snr.append(snr_db)
        results_cs_adap.append(np.mean(cs_list_adap))
        results_cs_fixed.append(np.mean(cs_list_fixed))
        results_sop_adap.append(outage_adap / num_samples)
        results_sop_fixed.append(outage_fixed / num_samples)
        results_phi_avg.append(np.mean(phi_star_list))
        
        print(f"{snr_db:>7} | {results_cs_adap[-1]:>12.4f} | {results_cs_fixed[-1]:>12.4f} | {results_sop_adap[-1]:>10.4f} | {results_sop_fixed[-1]:>10.4f}")

    # === 시각화 ===
    plt.figure(figsize=(18, 5))

    # 1. Capacity
    plt.subplot(1, 3, 1)
    plt.plot(results_snr, results_cs_adap, 's-', color='#3b82f6', label='Proposed (Adaptive)', linewidth=2)
    plt.plot(results_snr, results_cs_fixed, 'o--', color='#f59e0b', label='Fixed (phi=0.5)', linewidth=1.5)
    plt.xlabel('Average Transmit SNR (dB)')
    plt.ylabel('Secrecy Capacity (bps/Hz)')
    plt.title('Ergodic Secrecy Capacity')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # 2. Phi Trend
    plt.subplot(1, 3, 2)
    plt.plot(results_snr, results_phi_avg, 'D-', color='#8b5cf6', label='Optimal Phi*', linewidth=2)
    plt.xlabel('Average Transmit SNR (dB)')
    plt.ylabel('Optimal Power Ratio (phi*)')
    plt.title('Power Allocation Strategy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # 3. SOP
    plt.subplot(1, 3, 3)
    plt.semilogy(results_snr, results_sop_adap, 's-', color='#3b82f6', label='Proposed (Adaptive)', linewidth=2)
    plt.semilogy(results_snr, results_sop_fixed, 'o--', color='#f59e0b', label='Fixed (phi=0.5)', linewidth=1.5)
    plt.xlabel('Average Transmit SNR (dB)')
    plt.ylabel('SOP (Log Scale)')
    plt.title(f'Secrecy Outage Probability (Rs={Rs})')
    plt.grid(True, which="both", linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_main_simulation()
import numpy as np
import matplotlib.pyplot as plt

# === 뉴턴법 기반 최적화 함수 ===
def solve_optimal_phi_newton(P_total, sigma_n2, h, g, tol=1e-6, max_iter=15):
    norm_h = np.linalg.norm(h)
    w_s = h.conj().T / norm_h
    w_z = np.array([-h[1].conj(), h[0].conj()])
    w_z = w_z / np.linalg.norm(w_z)

    Kb = (P_total * np.abs(np.dot(h, w_s))**2) / sigma_n2
    Kes = (P_total * np.abs(np.dot(g, w_s))**2) / sigma_n2
    Kez = (P_total * np.abs(np.dot(g, w_z))**2) / sigma_n2

    phi = 0.5
    for i in range(max_iter):
        den_b = 1 + Kb * (1 - phi)
        den_e1 = 1 + Kes + (Kez - Kes) * phi
        den_e2 = 1 + Kez * phi
        
        f_prime = (-Kb / den_b) - ((Kez - Kes) / den_e1) + (Kez / den_e2)
        f_double_prime = (-Kb**2 / den_b**2) + ((Kez - Kes)**2 / den_e1**2) - (Kez**2 / den_e2**2)
        
        phi_next = phi - f_prime / f_double_prime
        phi_next = np.clip(phi_next, 0, 0.99)
        
        if np.abs(phi_next - phi) < tol:
            phi = phi_next
            break
        phi = phi_next
    
    Cb_final = np.log2(1 + Kb * (1 - phi))
    Ce_final = np.log2(1 + (Kes * (1 - phi)) / (Kez * phi + 1))
    cs_final = max(0, Cb_final - Ce_final)
    
    return phi, cs_final, Kb, Kes, Kez

# === 몬테카를로 시뮬레이션 및 SOP 계산 ===
def run_main_simulation():
    num_samples = 100000  # 샘플 수
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

    # 콘솔 헤더 출력
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
            
            # Adaptive Newton (Proposed)
            phi_star, cs_adap, Kb, Kes, Kez = solve_optimal_phi_newton(P_total, sigma_n2, h, g)
            
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
        
        # 수치 출력
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
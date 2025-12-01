"""
Single-Stage Membrane CO2 Capture Model
For post-combustion flue gas separation
Based on solution-diffusion permeation model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
GPU_TO_SI = 3.348e-10  # Conversion: 1 GPU = 3.348e-10 mol/(m²·s·Pa)
R_GAS = 8.314  # J/(mol·K)

class MembraneSeparation:
    """
    Single-stage membrane separation model for CO2 capture
    """
    
    def __init__(self, feed_composition, feed_pressure, permeate_pressure, 
                 temperature, co2_permeance_gpu, selectivity):
        """
        Initialize membrane separation parameters
        
        Parameters:
        -----------
        feed_composition : float
            CO2 mole fraction in feed (e.g., 0.15 for 15 vol%)
        feed_pressure : float
            Feed pressure (bar)
        permeate_pressure : float
            Permeate pressure (bar)
        temperature : float
            Operating temperature (K)
        co2_permeance_gpu : float
            CO2 permeance in GPU
        selectivity : float
            CO2/N2 selectivity (α = P_CO2 / P_N2)
        """
        self.z = feed_composition  # Feed CO2 mole fraction
        self.P_f = feed_pressure * 1e5  # Convert bar to Pa
        self.P_p = permeate_pressure * 1e5  # Convert bar to Pa
        self.T = temperature  # K
        
        # Membrane properties
        self.P_CO2 = co2_permeance_gpu * GPU_TO_SI  # mol/(m²·s·Pa)
        self.alpha = selectivity
        self.P_N2 = self.P_CO2 / self.alpha  # mol/(m²·s·Pa)
        
    def solve_single_stage(self, feed_flow):
        """
        Solve single-stage membrane separation
        
        Parameters:
        -----------
        feed_flow : float
            Feed molar flow rate (kmol/s)
        
        Returns:
        --------
        dict : Results containing stage-cut, compositions, flows, and area
        """
        F = feed_flow  # kmol/s
        
        def equations(vars):
            """
            System of equations for single-stage membrane
            vars = [theta, y] where:
            - theta: stage-cut (P/F)
            - y: CO2 mole fraction in permeate
            """
            theta, y = vars
            
            # Permeate and retentate flows
            P = theta * F
            R = F - P
            
            # Retentate composition from CO2 balance
            # F*z = R*x + P*y => x = (F*z - P*y)/R
            if R <= 0:
                return [1e10, 1e10]
            x = (F * self.z - P * y) / R
            
            # Check physical bounds
            if x < 0 or x > 1 or y < 0 or y > 1:
                return [1e10, 1e10]
            
            # Partial pressures
            p_CO2_f = x * self.P_f  # Approximation: use retentate composition
            p_CO2_p = y * self.P_p
            p_N2_f = (1 - x) * self.P_f
            p_N2_p = (1 - y) * self.P_p
            
            # Flux equations
            J_CO2 = self.P_CO2 * (p_CO2_f - p_CO2_p)
            J_N2 = self.P_N2 * (p_N2_f - p_N2_p)
            
            # Avoid division by zero
            if abs(J_CO2) < 1e-20 or abs(J_N2) < 1e-20:
                return [1e10, 1e10]
            
            # Flux ratio should equal composition ratio
            # J_CO2/J_N2 = (P*y)/(P*(1-y)) = y/(1-y)
            flux_ratio = J_CO2 / J_N2
            comp_ratio = y / (1 - y) if y < 1 else 1e10
            
            eq1 = flux_ratio - comp_ratio
            
            # Overall mass balance normalized
            eq2 = (F * self.z - R * x - P * y) / (F * self.z + 1e-10)
            
            return [eq1, eq2]
        
        # Initial guess
        theta_init = 0.3
        y_init = 0.5
        
        try:
            solution = fsolve(equations, [theta_init, y_init], full_output=True)
            theta, y = solution[0]
            info = solution[1]
            
            # Check convergence
            if info['fvec'][0]**2 + info['fvec'][1]**2 > 1e-6:
                print("Warning: Solution may not have converged")
            
            # Calculate results
            P = theta * F
            R = F - P
            x = (F * self.z - P * y) / R
            
            # CO2 recovery
            recovery = (P * y) / (F * self.z)
            
            # Membrane area calculation
            p_CO2_f = x * self.P_f
            p_CO2_p = y * self.P_p
            delta_p_CO2 = p_CO2_f - p_CO2_p
            
            n_CO2_perm = P * y * 1000  # Convert kmol/s to mol/s
            A = n_CO2_perm / (self.P_CO2 * delta_p_CO2)  # m²
            
            results = {
                'stage_cut': theta,
                'permeate_co2': y,
                'retentate_co2': x,
                'co2_recovery': recovery,
                'permeate_flow': P,  # kmol/s
                'retentate_flow': R,  # kmol/s
                'membrane_area': A,  # m²
                'co2_permeated': n_CO2_perm,  # mol/s
            }
            
            return results
            
        except Exception as e:
            print(f"Error solving membrane equations: {e}")
            return None
    
    def estimate_compressor_work(self, feed_flow, P_initial=1.0):
        """
        Estimate isothermal compressor work
        
        Parameters:
        -----------
        feed_flow : float
            Feed molar flow rate (kmol/s)
        P_initial : float
            Initial pressure before compression (bar)
        
        Returns:
        --------
        float : Compressor work (kW)
        """
        if self.P_f / 1e5 <= P_initial:
            return 0.0
        
        # Isothermal compression work: W = n*R*T*ln(P2/P1)
        n_dot = feed_flow * 1000  # mol/s
        P1 = P_initial * 1e5  # Pa
        P2 = self.P_f  # Pa
        
        W = n_dot * R_GAS * self.T * np.log(P2 / P1) / 1000  # kW
        return W


def sensitivity_analysis():
    """
    Perform sensitivity analysis on membrane performance
    """
    # Base case parameters
    feed_composition = 0.15  # 15 vol% CO2
    temperature = 298  # K
    feed_flow = 1.0  # kmol/s (basis)
    
    # Membrane properties
    membranes = {
        'Polaris': {'permeance': 3000, 'selectivity': 30},
        'Advanced': {'permeance': 2500, 'selectivity': 680}
    }
    
    # Pressure cases
    cases = [
        {'name': 'Baseline', 'P_feed': 1.0, 'P_perm': 0.1},
        {'name': 'Compressed Feed', 'P_feed': 3.0, 'P_perm': 0.2},
        {'name': 'Atmospheric Permeate', 'P_feed': 1.0, 'P_perm': 1.0}
    ]
    
    print("=" * 80)
    print("MEMBRANE CO2 CAPTURE - SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nFeed Composition: {feed_composition*100:.1f} vol% CO2")
    print(f"Temperature: {temperature} K")
    print(f"Feed Flow (basis): {feed_flow} kmol/s\n")
    
    all_results = {}
    
    for membrane_name, props in membranes.items():
        print(f"\n{'='*80}")
        print(f"MEMBRANE: {membrane_name}")
        print(f"Permeance: {props['permeance']} GPU, Selectivity: {props['selectivity']}")
        print(f"{'='*80}\n")
        
        membrane_results = {}
        
        for case in cases:
            print(f"Case: {case['name']}")
            print(f"  Feed Pressure: {case['P_feed']} bar, Permeate Pressure: {case['P_perm']} bar")
            
            # Create membrane object
            membrane = MembraneSeparation(
                feed_composition=feed_composition,
                feed_pressure=case['P_feed'],
                permeate_pressure=case['P_perm'],
                temperature=temperature,
                co2_permeance_gpu=props['permeance'],
                selectivity=props['selectivity']
            )
            
            # Solve
            results = membrane.solve_single_stage(feed_flow)
            
            if results:
                # Calculate compressor work
                comp_work = membrane.estimate_compressor_work(feed_flow, P_initial=1.0)
                
                print(f"  Stage-cut: {results['stage_cut']:.3f}")
                print(f"  Permeate CO2: {results['permeate_co2']*100:.2f} %")
                print(f"  CO2 Recovery: {results['co2_recovery']*100:.2f} %")
                print(f"  Membrane Area: {results['membrane_area']:.2f} m²")
                print(f"  Compressor Work: {comp_work:.2f} kW")
                
                # Check targets
                target_purity = 0.80
                target_recovery = 0.80
                meets_purity = results['permeate_co2'] >= target_purity
                meets_recovery = results['co2_recovery'] >= target_recovery
                
                print(f"  Meets 80% purity target: {'YES' if meets_purity else 'NO'}")
                print(f"  Meets 80% recovery target: {'YES' if meets_recovery else 'NO'}")
                print()
                
                membrane_results[case['name']] = results
            else:
                print("  ERROR: Could not solve\n")
        
        all_results[membrane_name] = membrane_results
    
    return all_results


def plot_recovery_vs_area():
    """
    Generate plots of CO2 recovery vs membrane area
    """
    feed_composition = 0.15
    temperature = 298
    feed_flow = 1.0
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    membranes = {
        'Polaris': {'permeance': 3000, 'selectivity': 30, 'color': 'blue'},
        'Advanced': {'permeance': 2500, 'selectivity': 680, 'color': 'red'}
    }
    
    # Vary feed pressure
    feed_pressures = np.linspace(1.0, 5.0, 20)
    permeate_pressure = 0.1
    
    for membrane_name, props in membranes.items():
        recoveries = []
        areas = []
        purities = []
        
        for P_feed in feed_pressures:
            membrane = MembraneSeparation(
                feed_composition=feed_composition,
                feed_pressure=P_feed,
                permeate_pressure=permeate_pressure,
                temperature=temperature,
                co2_permeance_gpu=props['permeance'],
                selectivity=props['selectivity']
            )
            
            results = membrane.solve_single_stage(feed_flow)
            if results:
                recoveries.append(results['co2_recovery'] * 100)
                areas.append(results['membrane_area'])
                purities.append(results['permeate_co2'] * 100)
        
        # Plot recovery vs area
        axes[0].plot(areas, recoveries, '-o', label=membrane_name, 
                    color=props['color'], linewidth=2)
        
        # Plot purity vs area
        axes[1].plot(areas, purities, '-o', label=membrane_name, 
                    color=props['color'], linewidth=2)
    
    # Format plots
    axes[0].set_xlabel('Membrane Area (m²)', fontsize=12)
    axes[0].set_ylabel('CO₂ Recovery (%)', fontsize=12)
    axes[0].set_title('CO₂ Recovery vs Membrane Area', fontsize=14, fontweight='bold')
    axes[0].axhline(y=80, color='green', linestyle='--', label='80% Target')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Membrane Area (m²)', fontsize=12)
    axes[1].set_ylabel('Permeate CO₂ Purity (%)', fontsize=12)
    axes[1].set_title('CO₂ Purity vs Membrane Area', fontsize=14, fontweight='bold')
    axes[1].axhline(y=80, color='green', linestyle='--', label='80% Target')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('membrane_performance.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'membrane_performance.png'")
    plt.show()


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("SINGLE-STAGE MEMBRANE CO2 CAPTURE MODEL")
    print("="*80 + "\n")
    
    # Run sensitivity analysis
    results = sensitivity_analysis()
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")
    plot_recovery_vs_area()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

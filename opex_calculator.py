"""
COMPREHENSIVE OPEX (Operating Expenditure) CALCULATOR
For Membrane-Based CO₂ Capture Systems

Includes:
- Energy costs (compression, vacuum)
- Membrane replacement
- Maintenance and labor
- Utilities
- Consumables
"""

import numpy as np
from membrane_separation import MembraneSeparation, GPU_TO_SI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class OPEXCalculator:
    """Calculate detailed operating expenses for membrane CO₂ capture"""
    
    def __init__(self):
        # Cost assumptions (default values - can be adjusted)
        self.electricity_cost = 0.07  # $/kWh
        self.membrane_cost_per_m2 = 50  # $/m²
        self.membrane_lifetime = 5  # years
        self.labor_cost_per_hour = 50  # $/hour
        self.maintenance_factor = 0.03  # 3% of CAPEX per year
        self.operating_hours_per_year = 8000  # hours/year
        self.plant_availability = 0.92  # 92% uptime
        
        # Additional costs
        self.water_cost = 0.5  # $/m³
        self.chemicals_cost = 5000  # $/year (cleaning agents, etc.)
        self.insurance_rate = 0.01  # 1% of CAPEX per year
        self.admin_overhead = 0.15  # 15% of direct labor
        
    def calculate_compression_energy(self, feed_flow_kmol_s, P_initial, P_final, 
                                     temperature=298, efficiency=0.75, stages=1):
        """
        Calculate compression energy requirement
        
        Parameters:
        -----------
        feed_flow_kmol_s : float
            Feed flow rate (kmol/s)
        P_initial : float
            Initial pressure (bar)
        P_final : float
            Final pressure (bar)
        temperature : float
            Temperature (K)
        efficiency : float
            Compressor isentropic efficiency
        stages : int
            Number of compression stages
        
        Returns:
        --------
        power_kW : float
            Power requirement (kW)
        """
        R = 8.314  # J/mol/K
        gamma = 1.4  # Heat capacity ratio for air/flue gas
        
        # Convert to SI
        P1 = P_initial * 1e5  # Pa
        P2 = P_final * 1e5    # Pa
        n_dot = feed_flow_kmol_s * 1000  # mol/s
        
        # Pressure ratio per stage
        pressure_ratio = (P2 / P1) ** (1 / stages)
        
        # Isentropic work per stage
        W_stage = (gamma / (gamma - 1)) * R * temperature * \
                  (pressure_ratio ** ((gamma - 1) / gamma) - 1)
        
        # Total work (all stages, actual efficiency)
        W_total = stages * W_stage * n_dot / efficiency  # W
        
        return W_total / 1000  # kW
    
    def calculate_vacuum_energy(self, permeate_flow_kmol_s, P_permeate, P_ambient=1.0,
                               temperature=298, efficiency=0.70):
        """
        Calculate vacuum pump energy requirement
        
        Parameters:
        -----------
        permeate_flow_kmol_s : float
            Permeate flow rate (kmol/s)
        P_permeate : float
            Permeate pressure (bar)
        P_ambient : float
            Ambient pressure (bar)
        temperature : float
            Temperature (K)
        efficiency : float
            Vacuum pump efficiency
        
        Returns:
        --------
        power_kW : float
            Power requirement (kW)
        """
        if P_permeate >= P_ambient:
            return 0.0  # No vacuum needed
        
        R = 8.314  # J/mol/K
        P1 = P_permeate * 1e5  # Pa
        P2 = P_ambient * 1e5    # Pa
        n_dot = permeate_flow_kmol_s * 1000  # mol/s
        
        # Isothermal compression work (conservative for vacuum)
        W = R * temperature * n_dot * np.log(P2 / P1) / efficiency
        
        return W / 1000  # kW
    
    def calculate_annual_opex(self, membrane_area, compression_power, vacuum_power=0,
                             labor_hours_per_day=2, water_usage_m3_per_day=1):
        """
        Calculate total annual OPEX
        
        Parameters:
        -----------
        membrane_area : float
            Total membrane area (m²)
        compression_power : float
            Compressor power (kW)
        vacuum_power : float
            Vacuum pump power (kW)
        labor_hours_per_day : float
            Daily operator hours
        water_usage_m3_per_day : float
            Daily water usage (m³/day)
        
        Returns:
        --------
        opex_breakdown : dict
            Detailed OPEX breakdown
        """
        # Operating days per year
        operating_days = self.operating_hours_per_year / 24
        
        # 1. Energy Costs
        total_power = compression_power + vacuum_power  # kW
        annual_energy = total_power * self.operating_hours_per_year  # kWh/year
        energy_cost = annual_energy * self.electricity_cost  # $/year
        
        # 2. Membrane Replacement
        annual_membrane_replacement = (membrane_area * self.membrane_cost_per_m2) / self.membrane_lifetime
        
        # 3. Labor Costs
        annual_labor_hours = labor_hours_per_day * 365
        direct_labor_cost = annual_labor_hours * self.labor_cost_per_hour
        admin_labor_cost = direct_labor_cost * self.admin_overhead
        total_labor_cost = direct_labor_cost + admin_labor_cost
        
        # 4. Maintenance & Repairs
        estimated_capex = membrane_area * self.membrane_cost_per_m2 + \
                         compression_power * 500  # Rough CAPEX estimate
        maintenance_cost = estimated_capex * self.maintenance_factor
        
        # 5. Utilities (Water)
        annual_water_usage = water_usage_m3_per_day * 365
        water_cost = annual_water_usage * self.water_cost
        
        # 6. Chemicals & Consumables
        chemicals_cost = self.chemicals_cost
        
        # 7. Insurance
        insurance_cost = estimated_capex * self.insurance_rate
        
        # 8. Miscellaneous (2% of other costs)
        subtotal = (energy_cost + annual_membrane_replacement + total_labor_cost +
                   maintenance_cost + water_cost + chemicals_cost + insurance_cost)
        miscellaneous = subtotal * 0.02
        
        # Total OPEX
        total_opex = subtotal + miscellaneous
        
        # Availability adjustment
        effective_opex = total_opex / self.plant_availability
        
        # Build breakdown dictionary
        opex_breakdown = {
            'Energy': {
                'Annual Energy (kWh/year)': annual_energy,
                'Power (kW)': total_power,
                'Cost ($/year)': energy_cost,
                'Percentage': (energy_cost / total_opex * 100)
            },
            'Membrane Replacement': {
                'Area (m²)': membrane_area,
                'Lifetime (years)': self.membrane_lifetime,
                'Cost ($/year)': annual_membrane_replacement,
                'Percentage': (annual_membrane_replacement / total_opex * 100)
            },
            'Labor': {
                'Direct Labor ($/year)': direct_labor_cost,
                'Admin/Overhead ($/year)': admin_labor_cost,
                'Total Cost ($/year)': total_labor_cost,
                'Percentage': (total_labor_cost / total_opex * 100)
            },
            'Maintenance & Repairs': {
                'Cost ($/year)': maintenance_cost,
                'Percentage': (maintenance_cost / total_opex * 100)
            },
            'Utilities (Water)': {
                'Annual Usage (m³/year)': annual_water_usage,
                'Cost ($/year)': water_cost,
                'Percentage': (water_cost / total_opex * 100)
            },
            'Chemicals & Consumables': {
                'Cost ($/year)': chemicals_cost,
                'Percentage': (chemicals_cost / total_opex * 100)
            },
            'Insurance': {
                'Cost ($/year)': insurance_cost,
                'Percentage': (insurance_cost / total_opex * 100)
            },
            'Miscellaneous': {
                'Cost ($/year)': miscellaneous,
                'Percentage': (miscellaneous / total_opex * 100)
            },
            'Total OPEX': {
                'Annual ($/year)': total_opex,
                'With Availability Adjustment ($/year)': effective_opex,
                'Daily ($/day)': effective_opex / 365,
                'Hourly ($/hour)': effective_opex / 8760
            }
        }
        
        return opex_breakdown
    
    def calculate_co2_capture_cost(self, opex_breakdown, co2_captured_tonnes_per_year,
                                   capex_total, amortization_years=10, interest_rate=0.05):
        """
        Calculate levelized cost of CO₂ capture
        
        Parameters:
        -----------
        opex_breakdown : dict
            OPEX breakdown from calculate_annual_opex()
        co2_captured_tonnes_per_year : float
            Annual CO₂ captured (tonnes/year)
        capex_total : float
            Total capital expenditure ($)
        amortization_years : int
            CAPEX amortization period (years)
        interest_rate : float
            Discount rate
        
        Returns:
        --------
        cost_metrics : dict
            Various cost metrics
        """
        # Annual OPEX
        annual_opex = opex_breakdown['Total OPEX']['With Availability Adjustment ($/year)']
        
        # Annualized CAPEX (capital recovery factor)
        if interest_rate > 0:
            CRF = (interest_rate * (1 + interest_rate)**amortization_years) / \
                  ((1 + interest_rate)**amortization_years - 1)
        else:
            CRF = 1 / amortization_years
        
        annualized_capex = capex_total * CRF
        
        # Total annual cost
        total_annual_cost = annual_opex + annualized_capex
        
        # Cost per tonne CO₂
        cost_per_tonne = total_annual_cost / co2_captured_tonnes_per_year if co2_captured_tonnes_per_year > 0 else np.inf
        
        # Cost per kg CO₂
        cost_per_kg = cost_per_tonne / 1000
        
        # Additional metrics
        cost_metrics = {
            'Annual OPEX ($/year)': annual_opex,
            'Annualized CAPEX ($/year)': annualized_capex,
            'Total Annual Cost ($/year)': total_annual_cost,
            'CO₂ Captured (tonnes/year)': co2_captured_tonnes_per_year,
            'Cost per tonne CO₂ ($/tonne)': cost_per_tonne,
            'Cost per kg CO₂ ($/kg)': cost_per_kg,
            'OPEX Fraction': annual_opex / total_annual_cost,
            'CAPEX Fraction': annualized_capex / total_annual_cost
        }
        
        return cost_metrics
    
    def print_opex_report(self, opex_breakdown, cost_metrics=None):
        """Print formatted OPEX report"""
        
        print("\n" + "="*80)
        print("OPERATING EXPENDITURE (OPEX) ANALYSIS")
        print("Membrane-Based CO₂ Capture System")
        print("="*80)
        
        print("\nASSUMPTIONS:")
        print(f"  Electricity Cost:           ${self.electricity_cost:.3f}/kWh")
        print(f"  Membrane Cost:              ${self.membrane_cost_per_m2:.2f}/m²")
        print(f"  Membrane Lifetime:          {self.membrane_lifetime} years")
        print(f"  Labor Rate:                 ${self.labor_cost_per_hour:.2f}/hour")
        print(f"  Operating Hours:            {self.operating_hours_per_year:,} hours/year")
        print(f"  Plant Availability:         {self.plant_availability*100:.1f}%")
        
        print("\n" + "-"*80)
        print("ANNUAL OPEX BREAKDOWN:")
        print("-"*80)
        
        categories = ['Energy', 'Membrane Replacement', 'Labor', 'Maintenance & Repairs',
                     'Utilities (Water)', 'Chemicals & Consumables', 'Insurance', 'Miscellaneous']
        
        for category in categories:
            if category in opex_breakdown:
                data = opex_breakdown[category]
                
                # Handle Labor category differently (has Total Cost instead of Cost)
                if category == 'Labor':
                    cost = data.get('Total Cost ($/year)', 0)
                else:
                    cost = data.get('Cost ($/year)', 0)
                
                pct = data.get('Percentage', 0)
                print(f"\n{category}:")
                print(f"  Cost:      ${cost:,.2f}/year  ({pct:.1f}%)")
                
                # Print additional details
                for key, value in data.items():
                    if key not in ['Cost ($/year)', 'Total Cost ($/year)', 'Percentage']:
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:,.2f}")
        
        print("\n" + "-"*80)
        print("TOTAL ANNUAL OPEX:")
        print("-"*80)
        total_data = opex_breakdown['Total OPEX']
        print(f"  Base Annual OPEX:           ${total_data['Annual ($/year)']:,.2f}/year")
        print(f"  With Availability Factor:   ${total_data['With Availability Adjustment ($/year)']:,.2f}/year")
        print(f"  Daily OPEX:                 ${total_data['Daily ($/day)']:,.2f}/day")
        print(f"  Hourly OPEX:                ${total_data['Hourly ($/hour)']:,.2f}/hour")
        
        if cost_metrics:
            print("\n" + "="*80)
            print("LEVELIZED COST OF CO₂ CAPTURE:")
            print("="*80)
            print(f"  Annual OPEX:                ${cost_metrics['Annual OPEX ($/year)']:,.2f}/year")
            print(f"  Annualized CAPEX:           ${cost_metrics['Annualized CAPEX ($/year)']:,.2f}/year")
            print(f"  Total Annual Cost:          ${cost_metrics['Total Annual Cost ($/year)']:,.2f}/year")
            print(f"\n  CO₂ Captured:               {cost_metrics['CO₂ Captured (tonnes/year)']:,.2f} tonnes/year")
            print(f"\n  💰 Cost per tonne CO₂:      ${cost_metrics['Cost per tonne CO₂ ($/tonne)']:,.2f}/tonne")
            print(f"  💰 Cost per kg CO₂:         ${cost_metrics['Cost per kg CO₂ ($/kg)']:,.4f}/kg")
            print(f"\n  OPEX Contribution:          {cost_metrics['OPEX Fraction']*100:.1f}%")
            print(f"  CAPEX Contribution:         {cost_metrics['CAPEX Fraction']*100:.1f}%")
        
        print("\n" + "="*80 + "\n")


def calculate_case_study_opex():
    """Run OPEX calculation for a specific case study"""
    
    print("\n🔬 MEMBRANE CO₂ CAPTURE - OPEX CALCULATION")
    print("="*80)
    
    # Initialize calculator
    calc = OPEXCalculator()
    
    # Case study parameters
    print("\nCASE STUDY PARAMETERS:")
    print("-"*80)
    
    feed_flow = 1.0  # kmol/s
    feed_co2 = 0.15  # 15% CO₂
    temperature = 298  # K
    feed_pressure = 3.0  # bar
    permeate_pressure = 0.2  # bar
    
    print(f"  Feed Flow Rate:             {feed_flow} kmol/s")
    print(f"  Feed CO₂ Composition:       {feed_co2*100:.1f}%")
    print(f"  Temperature:                {temperature} K")
    print(f"  Feed Pressure:              {feed_pressure} bar")
    print(f"  Permeate Pressure:          {permeate_pressure} bar")
    
    # Calculate for both membrane types
    membrane_types = [
        {'name': 'Polaris™', 'permeance': 3000, 'selectivity': 30},
        {'name': 'Advanced', 'permeance': 2500, 'selectivity': 680}
    ]
    
    results_all = {}
    
    for mem_type in membrane_types:
        print(f"\n{'='*80}")
        print(f"MEMBRANE TYPE: {mem_type['name']}")
        print(f"  Permeance: {mem_type['permeance']} GPU, Selectivity: {mem_type['selectivity']}")
        print('='*80)
        
        # Create membrane object
        membrane = MembraneSeparation(
            feed_composition=feed_co2,
            feed_pressure=feed_pressure,
            permeate_pressure=permeate_pressure,
            temperature=temperature,
            co2_permeance_gpu=mem_type['permeance'],
            selectivity=mem_type['selectivity']
        )
        
        # Solve
        results = membrane.solve_single_stage(feed_flow)
        
        if not results:
            print("❌ Solution failed to converge")
            continue
        
        print(f"\nPERFORMANCE RESULTS:")
        print(f"  Stage-Cut:                  {results['stage_cut']:.4f}")
        print(f"  Permeate CO₂:               {results['permeate_co2']*100:.2f}%")
        print(f"  Retentate CO₂:              {results['retentate_co2']*100:.2f}%")
        print(f"  CO₂ Recovery:               {results['co2_recovery']*100:.2f}%")
        print(f"  Membrane Area:              {results['membrane_area']:.2f} m²")
        
        # Calculate compression energy
        comp_power = calc.calculate_compression_energy(
            feed_flow, P_initial=1.0, P_final=feed_pressure,
            temperature=temperature, efficiency=0.75, stages=1
        )
        
        # Calculate vacuum energy
        vacuum_power = calc.calculate_vacuum_energy(
            results['permeate_flow'], P_permeate=permeate_pressure,
            P_ambient=1.0, temperature=temperature, efficiency=0.70
        )
        
        print(f"  Compression Power:          {comp_power:.2f} kW")
        print(f"  Vacuum Power:               {vacuum_power:.2f} kW")
        print(f"  Total Power:                {comp_power + vacuum_power:.2f} kW")
        
        # Calculate OPEX
        opex_breakdown = calc.calculate_annual_opex(
            membrane_area=results['membrane_area'],
            compression_power=comp_power,
            vacuum_power=vacuum_power,
            labor_hours_per_day=2,
            water_usage_m3_per_day=1
        )
        
        # Calculate CO₂ captured
        co2_flow = feed_flow * feed_co2  # kmol/s
        co2_captured_flow = co2_flow * results['co2_recovery']  # kmol/s
        co2_captured_annual = co2_captured_flow * 44 * 3600 * calc.operating_hours_per_year / 1000  # tonnes/year
        
        # Estimate CAPEX
        capex_total = results['membrane_area'] * calc.membrane_cost_per_m2 + \
                     comp_power * 1000 + vacuum_power * 500  # Simple estimate
        
        # Calculate levelized cost
        cost_metrics = calc.calculate_co2_capture_cost(
            opex_breakdown, co2_captured_annual, capex_total,
            amortization_years=10, interest_rate=0.05
        )
        
        # Print report
        calc.print_opex_report(opex_breakdown, cost_metrics)
        
        # Store results
        results_all[mem_type['name']] = {
            'performance': results,
            'opex': opex_breakdown,
            'cost_metrics': cost_metrics,
            'power': comp_power + vacuum_power
        }
    
    return results_all, calc


def plot_opex_analysis(results_all):
    """Create comprehensive OPEX visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    membrane_types = list(results_all.keys())
    colors = ['#3498db', '#e74c3c']
    
    # Plot 1: OPEX Breakdown - Polaris
    ax1 = fig.add_subplot(gs[0, 0])
    if len(membrane_types) > 0:
        opex = results_all[membrane_types[0]]['opex']
        categories = ['Energy', 'Membrane Replacement', 'Labor', 'Maintenance & Repairs',
                     'Utilities (Water)', 'Chemicals & Consumables', 'Insurance', 'Miscellaneous']
        values = [opex[cat].get('Total Cost ($/year)', opex[cat].get('Cost ($/year)', 0)) / 1000 for cat in categories]
        ax1.pie(values, labels=[c.replace(' & ', '\n&\n') for c in categories], autopct='%1.1f%%',
               colors=plt.cm.Set3.colors, startangle=90)
        ax1.set_title(f'{membrane_types[0]} OPEX Breakdown', fontweight='bold', fontsize=12)
    
    # Plot 2: OPEX Breakdown - Advanced
    ax2 = fig.add_subplot(gs[0, 1])
    if len(membrane_types) > 1:
        opex = results_all[membrane_types[1]]['opex']
        categories = ['Energy', 'Membrane Replacement', 'Labor', 'Maintenance & Repairs',
                     'Utilities (Water)', 'Chemicals & Consumables', 'Insurance', 'Miscellaneous']
        values = [opex[cat].get('Total Cost ($/year)', opex[cat].get('Cost ($/year)', 0)) / 1000 for cat in categories]
        ax2.pie(values, labels=[c.replace(' & ', '\n&\n') for c in categories], autopct='%1.1f%%',
               colors=plt.cm.Set3.colors, startangle=90)
        ax2.set_title(f'{membrane_types[1]} OPEX Breakdown', fontweight='bold', fontsize=12)
    
    # Plot 3: Total Cost Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    total_costs = [results_all[mem]['cost_metrics']['Total Annual Cost ($/year)'] / 1000
                  for mem in membrane_types]
    bars = ax3.bar(membrane_types, total_costs, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Total Annual Cost (k$/year)', fontweight='bold')
    ax3.set_title('Total Annual Cost Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.1f}k', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Cost per Tonne CO₂
    ax4 = fig.add_subplot(gs[1, 0])
    costs_per_tonne = [results_all[mem]['cost_metrics']['Cost per tonne CO₂ ($/tonne)']
                      for mem in membrane_types]
    bars = ax4.bar(membrane_types, costs_per_tonne, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Cost ($/tonne CO₂)', fontweight='bold')
    ax4.set_title('CO₂ Capture Cost Comparison', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: OPEX vs CAPEX Contribution
    ax5 = fig.add_subplot(gs[1, 1])
    opex_fractions = [results_all[mem]['cost_metrics']['OPEX Fraction'] * 100
                     for mem in membrane_types]
    capex_fractions = [results_all[mem]['cost_metrics']['CAPEX Fraction'] * 100
                      for mem in membrane_types]
    x = np.arange(len(membrane_types))
    width = 0.35
    ax5.bar(x - width/2, opex_fractions, width, label='OPEX', color='orange', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, capex_fractions, width, label='CAPEX', color='purple', alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Contribution (%)', fontweight='bold')
    ax5.set_title('OPEX vs CAPEX Contribution', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(membrane_types)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Energy Consumption
    ax6 = fig.add_subplot(gs[1, 2])
    power_values = [results_all[mem]['power'] for mem in membrane_types]
    bars = ax6.bar(membrane_types, power_values, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Power (kW)', fontweight='bold')
    ax6.set_title('Total Power Consumption', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} kW', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: Category-wise OPEX Comparison
    ax7 = fig.add_subplot(gs[2, :])
    categories = ['Energy', 'Membrane\nReplacement', 'Labor', 'Maintenance\n& Repairs',
                 'Utilities', 'Chemicals', 'Insurance', 'Misc.']
    category_keys = ['Energy', 'Membrane Replacement', 'Labor', 'Maintenance & Repairs',
                    'Utilities (Water)', 'Chemicals & Consumables', 'Insurance', 'Miscellaneous']
    
    x = np.arange(len(categories))
    width = 0.35
    
    values1 = [results_all[membrane_types[0]]['opex'][cat].get('Total Cost ($/year)', results_all[membrane_types[0]]['opex'][cat].get('Cost ($/year)', 0)) / 1000
              for cat in category_keys]
    values2 = [results_all[membrane_types[1]]['opex'][cat].get('Total Cost ($/year)', results_all[membrane_types[1]]['opex'][cat].get('Cost ($/year)', 0)) / 1000
              for cat in category_keys] if len(membrane_types) > 1 else []
    
    ax7.bar(x - width/2, values1, width, label=membrane_types[0], color=colors[0], alpha=0.7, edgecolor='black')
    if values2:
        ax7.bar(x + width/2, values2, width, label=membrane_types[1], color=colors[1], alpha=0.7, edgecolor='black')
    
    ax7.set_ylabel('Annual Cost (k$/year)', fontweight='bold')
    ax7.set_title('Detailed OPEX Category Comparison', fontweight='bold', fontsize=14)
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories, fontsize=9)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle('COMPREHENSIVE OPEX ANALYSIS - Membrane CO₂ Capture System',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_file = 'results/opex_analysis_comprehensive.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"\n✅ OPEX analysis plot saved to: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    # Run OPEX calculations
    results_all, calculator = calculate_case_study_opex()
    
    # Create visualizations
    plot_opex_analysis(results_all)
    
    print("\n✅ OPEX ANALYSIS COMPLETE")
    print(f"📊 Results saved to: results/opex_analysis_comprehensive.png")
    print(f"📄 Detailed breakdown printed above")

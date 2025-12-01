"""
Advanced Membrane Separation Simulation Engine
Supports multiple simulation modes: single, sweep, grid, Monte Carlo, batch
Version 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
from membrane_separation import MembraneSeparation, GPU_TO_SI
from opex_calculator import OPEXCalculator


class SimulationEngine:
    """
    Advanced simulation engine for membrane CO2 capture with multiple modes
    """
    
    def __init__(self):
        """Initialize simulation engine"""
        self.opex_calc = OPEXCalculator()
        self.results_history = []
        
    def single_simulation(self, params: Dict) -> Dict:
        """
        Run a single simulation with given parameters
        
        Parameters:
        -----------
        params : dict
            {
                'feed_flow': float (kmol/s),
                'feed_composition': float (0-1),
                'temperature': float (K),
                'feed_pressure': float (bar),
                'permeate_pressure': float (bar),
                'co2_permeance_gpu': float,
                'selectivity': float,
                'electricity_cost': float ($/kWh),
                'membrane_cost_per_m2': float ($),
                'o2_composition': float (optional, 0-1)
            }
        
        Returns:
        --------
        dict : Comprehensive simulation results
        """
        # Extract parameters
        feed_flow = params.get('feed_flow', 1.0)
        feed_composition = params.get('feed_composition', 0.15)
        temperature = params.get('temperature', 298)
        feed_pressure = params.get('feed_pressure', 3.0)
        permeate_pressure = params.get('permeate_pressure', 0.2)
        co2_permeance_gpu = params.get('co2_permeance_gpu', 1000)
        selectivity = params.get('selectivity', 50)
        electricity_cost = params.get('electricity_cost', 0.07)
        membrane_cost_per_m2 = params.get('membrane_cost_per_m2', 50)
        o2_composition = params.get('o2_composition', 0.0)
        
        # Adjust feed composition if O2 is present
        if o2_composition > 0:
            # Normalize: CO2 + N2 + O2 = 1
            n2_composition = 1 - feed_composition - o2_composition
            if n2_composition < 0:
                raise ValueError("Invalid composition: CO2 + O2 > 1")
        else:
            n2_composition = 1 - feed_composition
            
        # Create membrane system
        membrane = MembraneSeparation(
            feed_composition=feed_composition,
            feed_pressure=feed_pressure,
            permeate_pressure=permeate_pressure,
            temperature=temperature,
            co2_permeance_gpu=co2_permeance_gpu,
            selectivity=selectivity
        )
        
        # Solve membrane separation
        results = membrane.solve_single_stage(feed_flow)
        
        # Calculate economics
        self.opex_calc.electricity_cost = electricity_cost
        self.opex_calc.membrane_cost_per_m2 = membrane_cost_per_m2
        
        # Compression energy
        compression_power = self.opex_calc.calculate_compression_energy(
            feed_flow_kmol_s=feed_flow,
            P_initial=1.0,
            P_final=feed_pressure,
            temperature=temperature
        )
        
        # Vacuum power
        vacuum_power = 0
        if permeate_pressure < 1.0:
            vacuum_power = self.opex_calc.calculate_compression_energy(
                feed_flow_kmol_s=results['permeate_flow'],
                P_initial=permeate_pressure,
                P_final=1.0,
                temperature=temperature
            )
        
        # Calculate OPEX
        opex_results = self.opex_calc.calculate_annual_opex(
            membrane_area=results['membrane_area'],
            compression_power=compression_power,
            vacuum_power=vacuum_power
        )
        
        # Calculate CAPEX
        membrane_capex = results['membrane_area'] * membrane_cost_per_m2
        total_power = opex_results['Energy']['Power (kW)']
        compressor_capex = total_power * 500  # $/kW
        module_housing_capex = results['membrane_area'] * 20
        installation_capex = (membrane_capex + compressor_capex) * 0.15
        engineering_capex = (membrane_capex + compressor_capex) * 0.10
        contingency_capex = (membrane_capex + compressor_capex + module_housing_capex) * 0.20
        
        total_capex = (membrane_capex + compressor_capex + module_housing_capex + 
                      installation_capex + engineering_capex + contingency_capex)
        
        capex_breakdown = {
            'Membranes': membrane_capex,
            'Compressor': compressor_capex,
            'Module Housing': module_housing_capex,
            'Installation': installation_capex,
            'Engineering': engineering_capex,
            'Contingency': contingency_capex,
            'Total': total_capex
        }
        
        # Cost per ton CO2
        co2_captured_mol_s = results['co2_permeated']
        co2_captured_kg_yr = (co2_captured_mol_s * 44 / 1000 * 
                             self.opex_calc.operating_hours_per_year * 3600)
        co2_captured_ton_yr = co2_captured_kg_yr / 1000
        
        if co2_captured_ton_yr > 0:
            cost_per_ton = opex_results['Total OPEX']['Annual ($/year)'] / co2_captured_ton_yr
        else:
            cost_per_ton = float('inf')
        
        # Simple payback
        annual_savings = 0  # Could be added if there's a baseline
        annual_opex = opex_results['Total OPEX']['Annual ($/year)']
        if annual_savings > annual_opex:
            simple_payback = total_capex / (annual_savings - annual_opex)
        else:
            simple_payback = float('inf')
        
        # Pressure ratio
        pressure_ratio = feed_pressure / permeate_pressure
        
        # Energy per ton CO2
        if co2_captured_ton_yr > 0:
            energy_kwh_per_ton = (total_power * self.opex_calc.operating_hours_per_year) / co2_captured_ton_yr
        else:
            energy_kwh_per_ton = float('inf')
        
        # DOE target check (80/80)
        doe_pass = (results['co2_recovery'] >= 0.80 and results['permeate_co2'] >= 0.80)
        
        # Compile comprehensive results
        comprehensive_results = {
            **results,
            'feed_pressure': feed_pressure,
            'permeate_pressure': permeate_pressure,
            'pressure_ratio': pressure_ratio,
            'feed_composition': feed_composition,
            'o2_composition': o2_composition,
            'n2_composition': n2_composition,
            'temperature': temperature,
            'feed_flow': feed_flow,
            'co2_permeance_gpu': co2_permeance_gpu,
            'selectivity': selectivity,
            'compression_power': compression_power,
            'vacuum_power': vacuum_power,
            'total_power': total_power,
            'opex': opex_results,
            'capex': capex_breakdown,
            'cost_per_ton_co2': cost_per_ton,
            'energy_kwh_per_ton': energy_kwh_per_ton,
            'co2_captured_ton_yr': co2_captured_ton_yr,
            'simple_payback_years': simple_payback,
            'doe_target_pass': doe_pass,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.results_history.append(comprehensive_results)
        
        return comprehensive_results
    
    def parameter_sweep(self, base_params: Dict, sweep_param: str, 
                       sweep_range: Tuple[float, float], num_points: int = 30) -> pd.DataFrame:
        """
        Sweep a single parameter and collect results
        
        Parameters:
        -----------
        base_params : dict
            Base simulation parameters
        sweep_param : str
            Parameter to sweep ('temperature', 'feed_pressure', 'feed_composition', 'o2_composition', etc.)
        sweep_range : tuple
            (min, max) for sweep
        num_points : int
            Number of points in sweep
        
        Returns:
        --------
        DataFrame : Results for all sweep points
        """
        sweep_values = np.linspace(sweep_range[0], sweep_range[1], num_points)
        results_list = []
        
        for value in sweep_values:
            params = base_params.copy()
            params[sweep_param] = value
            
            try:
                result = self.single_simulation(params)
                results_list.append(result)
            except Exception as e:
                print(f"Warning: Simulation failed at {sweep_param}={value}: {e}")
                continue
        
        # Convert to DataFrame
        df = self._results_to_dataframe(results_list)
        return df
    
    def grid_sweep(self, base_params: Dict, param1: str, range1: Tuple[float, float],
                   param2: str, range2: Tuple[float, float], 
                   num_points1: int = 20, num_points2: int = 20) -> pd.DataFrame:
        """
        2D parameter grid sweep
        
        Parameters:
        -----------
        base_params : dict
            Base simulation parameters
        param1, param2 : str
            Parameters to sweep
        range1, range2 : tuple
            (min, max) for each parameter
        num_points1, num_points2 : int
            Number of points for each dimension
        
        Returns:
        --------
        DataFrame : Grid sweep results
        """
        values1 = np.linspace(range1[0], range1[1], num_points1)
        values2 = np.linspace(range2[0], range2[1], num_points2)
        results_list = []
        
        for v1 in values1:
            for v2 in values2:
                params = base_params.copy()
                params[param1] = v1
                params[param2] = v2
                
                try:
                    result = self.single_simulation(params)
                    results_list.append(result)
                except Exception as e:
                    print(f"Warning: Simulation failed at {param1}={v1}, {param2}={v2}: {e}")
                    continue
        
        df = self._results_to_dataframe(results_list)
        return df
    
    def monte_carlo_simulation(self, base_params: Dict, 
                              uncertainty_params: Dict[str, Tuple[float, float]],
                              num_samples: int = 1000) -> pd.DataFrame:
        """
        Monte Carlo uncertainty analysis
        
        Parameters:
        -----------
        base_params : dict
            Base simulation parameters
        uncertainty_params : dict
            {param_name: (mean, std_dev)} for uncertain parameters
        num_samples : int
            Number of Monte Carlo samples
        
        Returns:
        --------
        DataFrame : Monte Carlo results with statistics
        """
        results_list = []
        
        for i in range(num_samples):
            params = base_params.copy()
            
            # Sample uncertain parameters from normal distribution
            for param, (mean, std) in uncertainty_params.items():
                params[param] = np.random.normal(mean, std)
                # Ensure positive values
                if params[param] < 0:
                    params[param] = abs(params[param])
            
            try:
                result = self.single_simulation(params)
                result['sample_id'] = i
                results_list.append(result)
            except Exception as e:
                print(f"Warning: Sample {i} failed: {e}")
                continue
        
        df = self._results_to_dataframe(results_list)
        return df
    
    def batch_scenario_comparison(self, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple predefined scenarios
        
        Parameters:
        -----------
        scenarios : dict
            {scenario_name: params_dict}
        
        Returns:
        --------
        DataFrame : Comparison results
        """
        results_list = []
        
        for scenario_name, params in scenarios.items():
            try:
                result = self.single_simulation(params)
                result['scenario_name'] = scenario_name
                results_list.append(result)
            except Exception as e:
                print(f"Warning: Scenario '{scenario_name}' failed: {e}")
                continue
        
        df = self._results_to_dataframe(results_list)
        return df
    
    def o2_injection_study(self, base_params: Dict, 
                          o2_range: Tuple[float, float] = (0, 0.10),
                          num_points: int = 20) -> pd.DataFrame:
        """
        Study effect of O2 injection on membrane performance
        
        Parameters:
        -----------
        base_params : dict
            Base parameters
        o2_range : tuple
            Range of O2 mole fraction to inject (0-1)
        num_points : int
            Number of points
        
        Returns:
        --------
        DataFrame : O2 injection study results
        """
        return self.parameter_sweep(base_params, 'o2_composition', o2_range, num_points)
    
    def thermal_ramp_study(self, base_params: Dict,
                          temp_range: Tuple[float, float] = (273, 373),
                          num_points: int = 30) -> pd.DataFrame:
        """
        Study temperature effects on membrane performance
        
        Parameters:
        -----------
        base_params : dict
            Base parameters
        temp_range : tuple
            Temperature range (K)
        num_points : int
            Number of points
        
        Returns:
        --------
        DataFrame : Temperature study results
        """
        return self.parameter_sweep(base_params, 'temperature', temp_range, num_points)
    
    def _results_to_dataframe(self, results_list: List[Dict]) -> pd.DataFrame:
        """Convert list of result dictionaries to DataFrame"""
        if not results_list:
            return pd.DataFrame()
        
        # Flatten nested dictionaries
        flattened_results = []
        for result in results_list:
            flat = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    # Skip nested dicts like opex breakdown for now
                    # Could expand if needed
                    continue
                else:
                    flat[key] = value
            flattened_results.append(flat)
        
        return pd.DataFrame(flattened_results)
    
    def export_results_csv(self, df: pd.DataFrame, filename: str):
        """Export results to CSV"""
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def export_results_json(self, results: Union[Dict, List[Dict]], filename: str):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results exported to {filename}")
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from simulation results
        
        Parameters:
        -----------
        df : DataFrame
            Simulation results
        
        Returns:
        --------
        dict : Summary statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'count': len(df),
            'means': df[numeric_cols].mean().to_dict(),
            'stds': df[numeric_cols].std().to_dict(),
            'mins': df[numeric_cols].min().to_dict(),
            'maxs': df[numeric_cols].max().to_dict(),
            'medians': df[numeric_cols].median().to_dict()
        }
        
        # DOE target pass rate
        if 'doe_target_pass' in df.columns:
            summary['doe_pass_rate'] = df['doe_target_pass'].mean()
        
        return summary


class AdvancedAnalytics:
    """
    Advanced analytics for membrane simulation results
    """
    
    @staticmethod
    def pareto_front(df: pd.DataFrame, 
                    objective1: str = 'membrane_area',
                    objective2: str = 'co2_recovery',
                    minimize_obj1: bool = True,
                    maximize_obj2: bool = True) -> pd.DataFrame:
        """
        Compute Pareto front for multi-objective optimization
        
        Parameters:
        -----------
        df : DataFrame
            Simulation results
        objective1, objective2 : str
            Column names for objectives
        minimize_obj1, maximize_obj2 : bool
            Whether to minimize or maximize each objective
        
        Returns:
        --------
        DataFrame : Pareto optimal solutions
        """
        # Create copy
        data = df[[objective1, objective2]].copy()
        
        # Adjust for minimization/maximization
        if not minimize_obj1:
            data[objective1] = -data[objective1]
        if not maximize_obj2:
            data[objective2] = -data[objective2]
        
        # Find Pareto front
        is_pareto = np.ones(len(data), dtype=bool)
        for i, row in enumerate(data.values):
            if is_pareto[i]:
                # Check if dominated by any other point
                is_pareto[is_pareto] = np.any(
                    data.values[is_pareto] > row, axis=1
                ) | (i == np.arange(len(data))[is_pareto])
        
        return df[is_pareto].copy()
    
    @staticmethod
    def knee_point_detection(x: np.ndarray, y: np.ndarray) -> int:
        """
        Detect knee point in curve using second derivative method
        
        Parameters:
        -----------
        x, y : ndarray
            Curve data
        
        Returns:
        --------
        int : Index of knee point
        """
        if len(x) < 3:
            return 0
        
        # Calculate second derivative
        first_deriv = np.diff(y) / np.diff(x)
        second_deriv = np.diff(first_deriv) / np.diff(x[:-1])
        
        # Find maximum absolute second derivative
        knee_idx = np.argmax(np.abs(second_deriv)) + 1
        
        return knee_idx
    
    @staticmethod
    def uncertainty_quantification(df: pd.DataFrame, metric: str) -> Dict:
        """
        Quantify uncertainty in a metric from Monte Carlo results
        
        Parameters:
        -----------
        df : DataFrame
            Monte Carlo results
        metric : str
            Metric to analyze
        
        Returns:
        --------
        dict : Uncertainty statistics
        """
        values = df[metric].dropna()
        
        return {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'p5': values.quantile(0.05),
            'p25': values.quantile(0.25),
            'p50': values.quantile(0.50),
            'p75': values.quantile(0.75),
            'p95': values.quantile(0.95),
            'coefficient_of_variation': values.std() / values.mean() if values.mean() != 0 else float('inf')
        }


if __name__ == "__main__":
    # Example usage
    engine = SimulationEngine()
    
    # Base parameters
    base_params = {
        'feed_flow': 1.0,
        'feed_composition': 0.15,
        'temperature': 298,
        'feed_pressure': 3.0,
        'permeate_pressure': 0.2,
        'co2_permeance_gpu': 1000,
        'selectivity': 50,
        'electricity_cost': 0.07,
        'membrane_cost_per_m2': 50
    }
    
    # Single simulation
    print("Running single simulation...")
    result = engine.single_simulation(base_params)
    print(f"Recovery: {result['co2_recovery']*100:.2f}%")
    print(f"Purity: {result['permeate_co2']*100:.2f}%")
    print(f"Cost per ton: ${result['cost_per_ton_co2']:.2f}")
    
    # Parameter sweep
    print("\nRunning pressure sweep...")
    sweep_results = engine.parameter_sweep(base_params, 'feed_pressure', (1, 10), 20)
    print(f"Sweep complete: {len(sweep_results)} points")
    
    # Export results
    engine.export_results_csv(sweep_results, 'pressure_sweep_results.csv')

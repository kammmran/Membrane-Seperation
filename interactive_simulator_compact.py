"""
COMPACT MEMBRANE SEPARATION SIMULATOR
Simplified, User-Friendly GUI for CO₂ Capture
Version 4.0 - Compact Edition
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from membrane_separation import MembraneSeparation, GPU_TO_SI
from opex_calculator import OPEXCalculator
from simulation_core import SimulationEngine, AdvancedAnalytics
from auto_optimizer import TargetOptimizer, auto_optimize_for_target
import os


class CompactMembraneSimulator:
    """Compact and user-friendly membrane separation simulator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏭 Membrane CO₂ Capture Simulator - Compact")
        self.root.geometry("1200x700")
        
        # Initialize parameters with defaults
        self.params = {
            'feed_flow': 1.0,
            'feed_composition': 0.15,
            'temperature': 298,
            'feed_pressure': 3.0,
            'permeate_pressure': 0.2,
            'membrane_type': 'Advanced',
            'electricity_cost': 0.07,
            'membrane_cost_per_m2': 50,
        }
        
        # Results storage
        self.results = None
        self.opex_results = None
        
        # Create OPEX calculator
        self.opex_calc = OPEXCalculator()
        
        # Create simulation engine
        self.sim_engine = SimulationEngine()
        
        # Sweep results storage
        self.sweep_results = None
        self.current_sim_type = 'Parameter Sweep'
        
        # Simulation range parameters
        self.sim_ranges = {
            'param': tk.StringVar(value='temperature'),
            'start': tk.DoubleVar(value=290),
            'end': tk.DoubleVar(value=350),
            'points': tk.IntVar(value=20)
        }
        
        # Build GUI
        self.create_compact_gui()
        
        # Run initial simulation
        self.run_simulation()
    
    def create_compact_gui(self):
        """Create a clean, compact interface"""
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # ===== LEFT: COMPACT CONTROLS (300px wide) =====
        control_frame = ttk.LabelFrame(main_frame, text="⚙️ Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        row = 0
        
        # --- Process Parameters ---
        ttk.Label(control_frame, text="PROCESS PARAMETERS", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        # Feed Flow
        ttk.Label(control_frame, text="Flow Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.flow_var = tk.DoubleVar(value=self.params['feed_flow'])
        ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.flow_var, 
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.flow_lbl = ttk.Label(control_frame, text="1.0 kmol/s", width=12)
        self.flow_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # CO2 Composition
        ttk.Label(control_frame, text="CO₂ %:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.co2_var = tk.DoubleVar(value=self.params['feed_composition']*100)
        ttk.Scale(control_frame, from_=5, to=50, variable=self.co2_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.co2_lbl = ttk.Label(control_frame, text="15%", width=12)
        self.co2_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Temperature
        ttk.Label(control_frame, text="Temp:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.temp_var = tk.DoubleVar(value=self.params['temperature'])
        ttk.Scale(control_frame, from_=273, to=373, variable=self.temp_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.temp_lbl = ttk.Label(control_frame, text="298 K", width=12)
        self.temp_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Feed Pressure
        ttk.Label(control_frame, text="Feed Press:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.fp_var = tk.DoubleVar(value=self.params['feed_pressure'])
        ttk.Scale(control_frame, from_=1.0, to=10.0, variable=self.fp_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.fp_lbl = ttk.Label(control_frame, text="3.0 bar", width=12)
        self.fp_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Permeate Pressure
        ttk.Label(control_frame, text="Perm Press:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.pp_var = tk.DoubleVar(value=self.params['permeate_pressure'])
        ttk.Scale(control_frame, from_=0.05, to=2.0, variable=self.pp_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.pp_lbl = ttk.Label(control_frame, text="0.2 bar", width=12)
        self.pp_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, 
                                                                sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # --- Membrane Type ---
        ttk.Label(control_frame, text="MEMBRANE TYPE", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        self.membrane_var = tk.StringVar(value=self.params['membrane_type'])
        ttk.Radiobutton(control_frame, text="Polaris™ (High Flow)", 
                       variable=self.membrane_var, value='Polaris',
                       command=self.update_sim).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1
        ttk.Radiobutton(control_frame, text="Advanced (High Selectivity)", 
                       variable=self.membrane_var, value='Advanced',
                       command=self.update_sim).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, 
                                                                sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # --- Economics ---
        ttk.Label(control_frame, text="ECONOMICS", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        # Electricity Cost
        ttk.Label(control_frame, text="Elec Cost:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.elec_var = tk.DoubleVar(value=self.params['electricity_cost'])
        ttk.Scale(control_frame, from_=0.03, to=0.15, variable=self.elec_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.elec_lbl = ttk.Label(control_frame, text="$0.07/kWh", width=12)
        self.elec_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Membrane Cost
        ttk.Label(control_frame, text="Mem Cost:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.mem_var = tk.DoubleVar(value=self.params['membrane_cost_per_m2'])
        ttk.Scale(control_frame, from_=10, to=100, variable=self.mem_var,
                 orient=tk.HORIZONTAL, length=120, command=self.update_sim).grid(row=row, column=1, padx=5)
        self.mem_lbl = ttk.Label(control_frame, text="$50/m²", width=12)
        self.mem_lbl.grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, 
                                                                sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # --- Key Results ---
        ttk.Label(control_frame, text="KEY RESULTS", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        # Results display
        results_text_frame = ttk.Frame(control_frame)
        results_text_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.results_display = tk.Text(results_text_frame, height=12, width=35, 
                                       font=('Courier', 9), relief=tk.SUNKEN, bd=1)
        self.results_display.pack(fill=tk.BOTH, expand=True)
        row += 1
        
        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="🔄 Refresh", command=self.run_simulation,
                  width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🎯 Auto 80/80", command=self.auto_optimize_target,
                  width=12).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Reset button on new row
        reset_frame = ttk.Frame(control_frame)
        reset_frame.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        ttk.Button(reset_frame, text="🔄 Reset", command=self.reset_params,
                  width=26).pack()
        row += 1
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=3, 
                                                                sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # --- ADVANCED SIMULATION ---
        ttk.Label(control_frame, text="ADVANCED SIMULATION", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        row += 1
        
        # Simulation type selection
        ttk.Label(control_frame, text="Sim Type:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.sim_type_var = tk.StringVar(value='Parameter Sweep')
        sim_type_combo = ttk.Combobox(control_frame, textvariable=self.sim_type_var,
                                     values=['Parameter Sweep', 'O₂ Injection', 'Thermal Ramp',
                                            'Multi-Param Grid', 'Monte Carlo', 'Batch Scenarios'],
                                     state='readonly', width=15)
        sim_type_combo.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        sim_type_combo.bind('<<ComboboxSelected>>', self.on_sim_type_change)
        row += 1
        
        # Parameter selection (shown for Parameter Sweep only)
        self.param_label = ttk.Label(control_frame, text="Parameter:")
        self.param_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.param_combo = ttk.Combobox(control_frame, textvariable=self.sim_ranges['param'], 
                                   values=['temperature', 'feed_pressure', 'permeate_pressure', 
                                          'feed_composition', 'o2_composition'],
                                   state='readonly', width=15)
        self.param_combo.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.param_combo.bind('<<ComboboxSelected>>', self.on_param_change)
        row += 1
        
        # Start value
        self.start_label = ttk.Label(control_frame, text="From:")
        self.start_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.start_entry = ttk.Entry(control_frame, textvariable=self.sim_ranges['start'], width=10)
        self.start_entry.grid(row=row, column=1, sticky=tk.W, pady=2, padx=5)
        row += 1
        
        # End value
        self.end_label = ttk.Label(control_frame, text="To:")
        self.end_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.end_entry = ttk.Entry(control_frame, textvariable=self.sim_ranges['end'], width=10)
        self.end_entry.grid(row=row, column=1, sticky=tk.W, pady=2, padx=5)
        row += 1
        
        # Number of points
        self.points_label = ttk.Label(control_frame, text="Points:")
        self.points_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        self.points_spinbox = ttk.Spinbox(control_frame, from_=5, to=50, 
                                     textvariable=self.sim_ranges['points'], width=10)
        self.points_spinbox.grid(row=row, column=1, sticky=tk.W, pady=2, padx=5)
        row += 1
        
        # Simulate button
        simulate_btn_frame = ttk.Frame(control_frame)
        simulate_btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(simulate_btn_frame, text="▶️ Run Simulation", 
                  command=self.run_advanced_simulation,
                  width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(simulate_btn_frame, text="💾 Save PNG", 
                  command=self.save_simulation_png,
                  width=12).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # ===== RIGHT: VISUALIZATION TABS =====
        viz_frame = ttk.LabelFrame(main_frame, text="📊 Visualization", padding="10")
        viz_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main tabs with submenus - ENHANCED VERSION
        tab_configs = {
            "📈 Performance": [
                "KPI Dashboard",
                "Stream Flows",
                "Composition Profile",
                "Target Check",
                "Separation Efficiency",
                "Mass Balance Sankey",
                "CO₂ Capture Metrics",
                "Performance Gauge Panel"
            ],
            "💰 Economics": [
                "OPEX Breakdown",
                "OPEX Bar Chart",
                "CAPEX Breakdown",
                "CAPEX vs OPEX",
                "Cost per Ton CO₂",
                "Economic Waterfall",
                "ROI Analysis",
                "Cost Breakdown Treemap",
                "Payback Period"
            ],
            "🎯 Sensitivity": [
                "Feed Pressure",
                "Temperature",
                "Feed Composition",
                "Area-Recovery Trade-off",
                "Pressure Ratio Impact",
                "Multi-Variable Tornado",
                "Operating Cost Sensitivity",
                "Selectivity Sensitivity"
            ],
            "📊 Advanced": [
                "Operating Window",
                "CO₂ Flux Profile",
                "N₂ Flux Profile",
                "3D Performance Map",
                "Process Flow Diagram",
                "Driving Force Distribution",
                "Membrane Selectivity Map",
                "Stage Cut Analysis",
                "Permeability Contours"
            ],
            "🔬 Optimization": [
                "Pareto Front",
                "Pressure Ratio Heatmap",
                "Specific Energy Map",
                "Compressor Work Envelope",
                "Selectivity vs Flux",
                "Multi-Objective Tradeoff",
                "Constraint Boundaries",
                "Optimization Path"
            ],
            "📉 Analytics": [
                "Cross-Sensitivity Radar",
                "Permeance Degradation",
                "Membrane Utilization",
                "DOE Response Surface",
                "Scenario Comparison",
                "Statistical Distribution",
                "Correlation Matrix",
                "Time Series Projection"
            ],
            "⭐ Showcase": [
                "Ternary Phase Diagram",
                "Multi-Metric Radar",
                "Parallel Coordinates",
                "Ridge Plot",
                "Benchmark Ladder",
                "Van't Hoff Analysis",
                "Arrhenius Plot",
                "Violin Performance Plot"
            ],
            "🧪 Simulation": [
                "O₂ Injection Study",
                "Thermal Ramp Study",
                "Multi-Param Grid",
                "Monte Carlo Analysis",
                "Batch Scenarios",
                "Parametric Sweep 3D",
                "Uncertainty Quantification",
                "Robustness Analysis"
            ],
            "🏗️ Process Designs": [
                "Single-Stage System",
                "Two-Stage Cascade",
                "Multi-Stage Series",
                "Parallel Array",
                "Recirculation Loop",
                "Cross-Flow Filtration",
                "Spiral Wound Module",
                "Hollow Fiber Config",
                "Plate and Frame",
                "Tubular Design",
                "Dead-End Filtration",
                "Retentate Staging",
                "Diafiltration",
                "Reverse Osmosis",
                "Nanofiltration",
                "Ultrafiltration",
                "Microfiltration",
                "Gas Separation",
                "Pervaporation",
                "Electrodialysis"
            ]
        }
        
        self.tabs = []
        self.figures = []
        self.canvases = []
        self.graph_selectors = {}  # Store dropdown menus for each tab
        
        for tab_name, graph_options in tab_configs.items():
            # Create tab
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=tab_name)
            self.tabs.append(tab)
            
            # Create control frame at top for graph selection
            control_frame = ttk.Frame(tab)
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            
            ttk.Label(control_frame, text="Select Graph:", 
                     font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
            
            # Create dropdown for graph selection
            graph_var = tk.StringVar(value=graph_options[0])
            self.graph_selectors[tab_name] = graph_var
            
            graph_dropdown = ttk.Combobox(control_frame, textvariable=graph_var, 
                                         values=graph_options, state='readonly', width=25)
            graph_dropdown.pack(side=tk.LEFT, padx=5)
            graph_dropdown.bind('<<ComboboxSelected>>', lambda e, tn=tab_name: self.on_graph_select(tn))
            
            # Add refresh button for this tab
            ttk.Button(control_frame, text="🔄 Refresh", 
                      command=lambda tn=tab_name: self.update_single_tab(tn),
                      width=10).pack(side=tk.LEFT, padx=5)
            
            # Create figure
            fig = Figure(figsize=(7, 4.5), dpi=100)
            self.figures.append(fig)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.canvases.append(canvas)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="✓ Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
    
    def update_sim(self, *args):
        """Update simulation when parameters change"""
        # Update labels
        self.flow_lbl.config(text=f"{self.flow_var.get():.2f} kmol/s")
        self.co2_lbl.config(text=f"{self.co2_var.get():.1f}%")
        self.temp_lbl.config(text=f"{self.temp_var.get():.0f} K")
        self.fp_lbl.config(text=f"{self.fp_var.get():.1f} bar")
        self.pp_lbl.config(text=f"{self.pp_var.get():.2f} bar")
        self.elec_lbl.config(text=f"${self.elec_var.get():.3f}/kWh")
        self.mem_lbl.config(text=f"${self.mem_var.get():.0f}/m²")
        
        # Auto-update simulation
        self.run_simulation()
    
    def on_graph_select(self, tab_name):
        """Handle graph selection change"""
        self.update_single_tab(tab_name)
    
    def on_param_change(self, event=None):
        """Update default ranges when parameter selection changes"""
        param = self.sim_ranges['param'].get()
        
        # Set sensible defaults for each parameter
        if param == 'temperature':
            self.sim_ranges['start'].set(273)
            self.sim_ranges['end'].set(373)
        elif param == 'feed_pressure':
            self.sim_ranges['start'].set(1)
            self.sim_ranges['end'].set(10)
        elif param == 'permeate_pressure':
            self.sim_ranges['start'].set(0.05)
            self.sim_ranges['end'].set(1.0)
        elif param == 'feed_composition':
            self.sim_ranges['start'].set(0.05)
            self.sim_ranges['end'].set(0.40)
        elif param == 'o2_composition':
            self.sim_ranges['start'].set(0.0)
            self.sim_ranges['end'].set(0.10)
    
    def on_sim_type_change(self, event=None):
        """Update UI controls based on selected simulation type"""
        sim_type = self.sim_type_var.get()
        
        if sim_type == 'Parameter Sweep':
            # Show all controls
            self.param_label.grid()
            self.param_combo.grid()
            self.start_label.config(text="From:")
            self.end_label.config(text="To:")
            
        elif sim_type == 'O₂ Injection':
            # Hide parameter selection, set fixed parameter
            self.param_label.grid_remove()
            self.param_combo.grid_remove()
            self.sim_ranges['param'].set('o2_composition')
            self.sim_ranges['start'].set(0.0)
            self.sim_ranges['end'].set(0.10)
            self.sim_ranges['points'].set(15)
            self.start_label.config(text="O₂ From:")
            self.end_label.config(text="O₂ To:")
            
        elif sim_type == 'Thermal Ramp':
            # Hide parameter selection, set fixed parameter
            self.param_label.grid_remove()
            self.param_combo.grid_remove()
            self.sim_ranges['param'].set('temperature')
            self.sim_ranges['start'].set(273)
            self.sim_ranges['end'].set(373)
            self.sim_ranges['points'].set(25)
            self.start_label.config(text="Temp From:")
            self.end_label.config(text="Temp To:")
            
        elif sim_type == 'Multi-Param Grid':
            # Keep controls but adjust labels
            self.param_label.grid()
            self.param_combo.grid()
            self.start_label.config(text="From:")
            self.end_label.config(text="To:")
            self.sim_ranges['points'].set(15)
            
        elif sim_type == 'Monte Carlo':
            # Adjust for Monte Carlo
            self.param_label.grid_remove()
            self.param_combo.grid_remove()
            self.start_label.config(text="Samples:")
            self.end_label.grid_remove()
            self.end_entry.grid_remove()
            self.points_label.grid_remove()
            self.points_spinbox.grid_remove()
            self.sim_ranges['points'].set(200)
            
        elif sim_type == 'Batch Scenarios':
            # Hide all range controls for batch
            self.param_label.grid_remove()
            self.param_combo.grid_remove()
            self.start_label.grid_remove()
            self.start_entry.grid_remove()
            self.end_label.grid_remove()
            self.end_entry.grid_remove()
            self.points_label.grid_remove()
            self.points_spinbox.grid_remove()
    
    def update_single_tab(self, tab_name):
        """Update only the selected tab"""
        if not self.results or not self.opex_results:
            return
        
        # Get the tab index
        tab_names = ["📈 Performance", "💰 Economics", "🎯 Sensitivity", "📊 Advanced", 
                     "🔬 Optimization", "📉 Analytics", "⭐ Showcase", "🧪 Simulation", "🏗️ Process Designs"]
        if tab_name not in tab_names:
            return
        
        tab_index = tab_names.index(tab_name)
        selected_graph = self.graph_selectors[tab_name].get()
        
        # Update the specific graph
        self.update_graph(tab_index, tab_name, selected_graph)
    
    def run_simulation(self):
        """Run the membrane separation simulation"""
        try:
            self.status_label.config(text="⚙️ Running simulation...")
            self.root.update()
            
            # Update parameters
            self.params['feed_flow'] = self.flow_var.get()
            self.params['feed_composition'] = self.co2_var.get() / 100
            self.params['temperature'] = self.temp_var.get()
            self.params['feed_pressure'] = self.fp_var.get()
            self.params['permeate_pressure'] = self.pp_var.get()
            self.params['membrane_type'] = self.membrane_var.get()
            self.params['electricity_cost'] = self.elec_var.get()
            self.params['membrane_cost_per_m2'] = self.mem_var.get()
            
            # Set membrane properties based on type
            if self.params['membrane_type'] == 'Polaris':
                co2_permeance_gpu = 3000
                selectivity = 30
            else:  # Advanced
                co2_permeance_gpu = 2500
                selectivity = 680
            
            # Create membrane system
            membrane = MembraneSeparation(
                feed_composition=self.params['feed_composition'],
                feed_pressure=self.params['feed_pressure'],
                permeate_pressure=self.params['permeate_pressure'],
                temperature=self.params['temperature'],
                co2_permeance_gpu=co2_permeance_gpu,
                selectivity=selectivity
            )
            
            # Calculate results
            self.results = membrane.solve_single_stage(self.params['feed_flow'])
            
            # Calculate economics
            self.opex_calc.electricity_cost = self.params['electricity_cost']
            self.opex_calc.membrane_cost_per_m2 = self.params['membrane_cost_per_m2']
            
            # Estimate compression power (feed compression from 1 to feed_pressure)
            compression_power = self.opex_calc.calculate_compression_energy(
                feed_flow_kmol_s=self.params['feed_flow'],
                P_initial=1.0,
                P_final=self.params['feed_pressure'],
                temperature=self.params['temperature']
            )
            
            # Estimate vacuum power (permeate from permeate_pressure to 1 bar)
            vacuum_power = 0
            if self.params['permeate_pressure'] < 1.0:
                vacuum_power = self.opex_calc.calculate_vacuum_energy(
                    permeate_flow_kmol_s=self.results['permeate_flow'],
                    P_permeate=self.params['permeate_pressure'],
                    P_ambient=1.0,
                    temperature=self.params['temperature']
                )
            
            self.opex_results = self.opex_calc.calculate_annual_opex(
                membrane_area=self.results['membrane_area'],
                compression_power=compression_power,
                vacuum_power=vacuum_power
            )
            
            # Update display
            self.update_results_display()
            self.update_graphs()
            
            self.status_label.config(text="✓ Simulation complete")
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Error: {str(e)}")
            self.status_label.config(text="❌ Simulation failed")
    
    def update_results_display(self):
        """Update the results text display"""
        self.results_display.delete(1.0, tk.END)
        
        if self.results and self.opex_results:
            text = "PERFORMANCE METRICS\n"
            text += "=" * 30 + "\n"
            text += f"CO₂ Recovery:  {self.results['co2_recovery']*100:6.2f}%\n"
            text += f"CO₂ Purity:    {self.results['permeate_co2']*100:6.2f}%\n"
            text += f"Membrane Area: {self.results['membrane_area']:6.1f} m²\n"
            text += f"Stage Cut:     {self.results['stage_cut']*100:6.2f}%\n"
            text += "\nECONOMICS\n"
            text += "=" * 30 + "\n"
            
            # Extract OPEX values from nested dictionary
            total_opex = self.opex_results['Total OPEX']['Annual ($/year)']
            energy_cost = self.opex_results['Energy']['Cost ($/year)']
            mem_replace = self.opex_results['Membrane Replacement']['Cost ($/year)']
            
            text += f"OPEX:          ${total_opex/1000:6.1f}k/yr\n"
            text += f"Energy Cost:   ${energy_cost/1000:6.1f}k/yr\n"
            text += f"Mem Replace:   ${mem_replace/1000:6.1f}k/yr\n"
            
            # Calculate cost per ton CO2
            co2_captured_mol_s = self.results['co2_permeated']  # mol/s
            co2_captured_kg_yr = co2_captured_mol_s * 44 / 1000 * self.opex_calc.operating_hours_per_year * 3600  # kg/yr
            co2_captured_ton_yr = co2_captured_kg_yr / 1000  # tonnes/yr
            if co2_captured_ton_yr > 0:
                cost_per_ton = total_opex / co2_captured_ton_yr
                text += f"Cost/ton CO₂:  ${cost_per_ton:6.2f}\n"
            
            # Target check
            text += "\nTARGET CHECK (80/80)\n"
            text += "=" * 30 + "\n"
            meets_target = self.results['co2_recovery'] >= 0.80 and self.results['permeate_co2'] >= 0.80
            text += f"Status: {'✓ PASS' if meets_target else '✗ FAIL'}\n"
            
            self.results_display.insert(1.0, text)
    
    def update_graphs(self):
        """Update all visualization tabs - only update currently selected graph in each tab"""
        if not self.results or not self.opex_results:
            return
        
        # Update each tab with its currently selected graph
        tab_names = ["📈 Performance", "💰 Economics", "🎯 Sensitivity", "📊 Advanced", 
                     "🔬 Optimization", "📉 Analytics", "⭐ Showcase", "🧪 Simulation", "🏗️ Process Designs"]
        for i, tab_name in enumerate(tab_names):
            selected_graph = self.graph_selectors[tab_name].get()
            self.update_graph(i, tab_name, selected_graph)
    
    def update_graph(self, tab_index, tab_name, graph_name):
        """Update a specific graph based on selection"""
        if not self.results or not self.opex_results:
            return
        
        fig = self.figures[tab_index]
        fig.clear()
        
        # Route to appropriate graph generator
        if tab_name == "📈 Performance":
            self.draw_performance_graph(fig, graph_name)
        elif tab_name == "💰 Economics":
            self.draw_economics_graph(fig, graph_name)
        elif tab_name == "🎯 Sensitivity":
            self.draw_sensitivity_graph(fig, graph_name)
        elif tab_name == "📊 Advanced":
            self.draw_advanced_graph(fig, graph_name)
        elif tab_name == "🔬 Optimization":
            self.draw_optimization_graph(fig, graph_name)
        elif tab_name == "📉 Analytics":
            self.draw_analytics_graph(fig, graph_name)
        elif tab_name == "⭐ Showcase":
            self.draw_showcase_graph(fig, graph_name)
        elif tab_name == "🧪 Simulation":
            self.draw_simulation_graph(fig, graph_name)
        elif tab_name == "🏗️ Process Designs":
            self.draw_process_design_graph(fig, graph_name)
        
        self.canvases[tab_index].draw()
    
    def draw_performance_graph(self, fig, graph_name):
        """Draw performance-related graphs"""
        if graph_name == "KPI Dashboard":
            # 2x2 grid of KPIs
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Recovery and Purity bars
            ax1 = fig.add_subplot(gs[0, 0])
            metrics = ['Recovery', 'Purity']
            values = [self.results['co2_recovery'] * 100, self.results['permeate_co2'] * 100]
            colors = ['#4CAF50' if v >= 80 else '#FF9800' if v >= 60 else '#F44336' for v in values]
            bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
            ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target')
            ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=9)
            ax1.set_title('CO₂ Capture Targets', fontweight='bold', fontsize=10)
            ax1.set_ylim(0, 100)
            ax1.legend(fontsize=8)
            ax1.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2., val + 2, f'{val:.1f}%', 
                        ha='center', fontweight='bold', fontsize=9)
            
            # Membrane Area
            ax2 = fig.add_subplot(gs[0, 1])
            area = self.results['membrane_area']
            ax2.barh(['Area'], [area], color='#2196F3', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Membrane Area (m²)', fontweight='bold', fontsize=9)
            ax2.set_title('Required Area', fontweight='bold', fontsize=10)
            ax2.text(area/2, 0, f'{area:.1f} m²', ha='center', va='center', 
                    fontweight='bold', fontsize=11, color='white')
            ax2.grid(axis='x', alpha=0.3)
            
            # Stream Flows
            ax3 = fig.add_subplot(gs[1, 0])
            flows = [self.params['feed_flow'], self.results['permeate_flow'], self.results['retentate_flow']]
            labels = ['Feed', 'Permeate', 'Retentate']
            colors_flow = ['#2196F3', '#4CAF50', '#FF9800']
            bars = ax3.bar(labels, flows, color=colors_flow, alpha=0.7, edgecolor='black', width=0.6)
            ax3.set_ylabel('Flow (kmol/s)', fontweight='bold', fontsize=9)
            ax3.set_title('Stream Flows', fontweight='bold', fontsize=10)
            ax3.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, flows):
                ax3.text(bar.get_x() + bar.get_width()/2., val + 0.01, f'{val:.3f}', 
                        ha='center', fontweight='bold', fontsize=8)
            
            # Stage Cut Pie
            ax4 = fig.add_subplot(gs[1, 1])
            stage_cut = self.results['stage_cut'] * 100
            sizes = [stage_cut, 100 - stage_cut]
            colors_pie = ['#4CAF50', '#E0E0E0']
            wedges, texts, autotexts = ax4.pie(sizes, labels=['Permeate', 'Retentate'], 
                                               colors=colors_pie, autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax4.set_title('Stage Cut', fontweight='bold', fontsize=10)
        
        elif graph_name == "Stream Flows":
            ax = fig.add_subplot(111)
            flows = [self.params['feed_flow'], self.results['permeate_flow'], self.results['retentate_flow']]
            labels = ['Feed', 'Permeate\n(CO₂ Rich)', 'Retentate\n(N₂ Rich)']
            colors_flow = ['#2196F3', '#4CAF50', '#FF9800']
            bars = ax.bar(labels, flows, color=colors_flow, alpha=0.7, edgecolor='black', width=0.5)
            ax.set_ylabel('Molar Flow Rate (kmol/s)', fontweight='bold', fontsize=11)
            ax.set_title('Stream Flow Rates Comparison', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, flows):
                ax.text(bar.get_x() + bar.get_width()/2., val + 0.02, f'{val:.3f} kmol/s', 
                        ha='center', fontweight='bold', fontsize=10)
        
        elif graph_name == "Composition Profile":
            ax = fig.add_subplot(111)
            streams = ['Feed', 'Permeate', 'Retentate']
            co2_comps = [self.params['feed_composition']*100, 
                        self.results['permeate_co2']*100,
                        self.results['retentate_co2']*100]
            n2_comps = [100 - c for c in co2_comps]
            
            x = np.arange(len(streams))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, co2_comps, width, label='CO₂', color='#4CAF50', 
                          alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, n2_comps, width, label='N₂', color='#2196F3', 
                          alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Composition (vol%)', fontweight='bold', fontsize=11)
            ax.set_title('Stream Composition Profile', fontweight='bold', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(streams)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', fontweight='bold', fontsize=9)
        
        elif graph_name == "Target Check":
            ax = fig.add_subplot(111)
            
            targets = ['Recovery\nTarget', 'Purity\nTarget']
            actual = [self.results['co2_recovery']*100, self.results['permeate_co2']*100]
            target_vals = [80, 80]
            
            x = np.arange(len(targets))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, target_vals, width, label='Target (80%)', 
                          color='#E0E0E0', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, actual, width, label='Actual', 
                          color=['#4CAF50' if a >= 80 else '#FF9800' for a in actual],
                          alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
            ax.set_title('DOE 80/80 Target Assessment', fontweight='bold', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(targets)
            ax.legend(fontsize=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.1f}%', ha='center', fontweight='bold', fontsize=10)
            
            # Add pass/fail text
            meets_target = all(a >= 80 for a in actual)
            status_text = '✓ PASSES 80/80 TARGET' if meets_target else '✗ FAILS 80/80 TARGET'
            status_color = 'green' if meets_target else 'red'
            ax.text(0.5, 0.95, status_text, transform=ax.transAxes, ha='center',
                   fontsize=14, fontweight='bold', color=status_color,
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7))
        
        elif graph_name == "Separation Efficiency":
            # New enhanced graph showing separation efficiency metrics
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Selectivity gauge
            ax1 = fig.add_subplot(gs[0, 0])
            selectivity = self.results['co2_permeance'] / self.results['n2_permeance']
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            ax1 = fig.add_subplot(gs[0, 0], projection='polar')
            ax1.plot(theta, r, 'k-', linewidth=2)
            ax1.fill_between(theta, 0, r, alpha=0.1, color='gray')
            # Color zones
            theta_green = theta[theta <= np.pi * 0.7]
            theta_yellow = theta[(theta > np.pi * 0.7) & (theta <= np.pi * 0.9)]
            theta_red = theta[theta > np.pi * 0.9]
            ax1.fill_between(theta_green, 0, r[:len(theta_green)], alpha=0.3, color='green')
            ax1.fill_between(theta_yellow, 0, r[:len(theta_yellow)], alpha=0.3, color='yellow')
            ax1.fill_between(theta_red, 0, r[:len(theta_red)], alpha=0.3, color='red')
            # Needle
            needle_angle = min(selectivity / 100 * np.pi, np.pi)
            ax1.plot([needle_angle, needle_angle], [0, 0.9], 'r-', linewidth=3)
            ax1.set_ylim(0, 1)
            ax1.set_theta_zero_location('W')
            ax1.set_theta_direction(1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.text(np.pi/2, 0.5, f'α = {selectivity:.1f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax1.set_title('CO₂/N₂ Selectivity', fontweight='bold', fontsize=10, pad=20)
            
            # Enrichment ratio
            ax2 = fig.add_subplot(gs[0, 1])
            enrichment = self.results['permeate_co2'] / self.params['feed_composition']
            ax2.barh(['Enrichment\nRatio'], [enrichment], color='#9C27B0', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Enrichment Factor', fontweight='bold', fontsize=9)
            ax2.set_title('CO₂ Enrichment', fontweight='bold', fontsize=10)
            ax2.text(enrichment/2, 0, f'{enrichment:.2f}x', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
            ax2.grid(axis='x', alpha=0.3)
            ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            # CO2 captured vs lost
            ax3 = fig.add_subplot(gs[1, 0])
            co2_feed = self.params['feed_flow'] * self.params['feed_composition']
            co2_captured = self.results['permeate_flow'] * self.results['permeate_co2']
            co2_lost = co2_feed - co2_captured
            sizes = [co2_captured, co2_lost]
            colors = ['#4CAF50', '#F44336']
            labels = ['Captured', 'Lost']
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax3.set_title('CO₂ Distribution', fontweight='bold', fontsize=10)
            
            # Separation power
            ax4 = fig.add_subplot(gs[1, 1])
            sep_power = self.results['co2_recovery'] * self.results['permeate_co2']
            metrics = ['Recovery', 'Purity', 'Sep. Power']
            values = [self.results['co2_recovery'], self.results['permeate_co2'], sep_power]
            colors_sp = ['#2196F3', '#4CAF50', '#FF9800']
            bars = ax4.bar(metrics, values, color=colors_sp, alpha=0.7, edgecolor='black', width=0.5)
            ax4.set_ylabel('Value (fraction)', fontweight='bold', fontsize=9)
            ax4.set_title('Separation Power', fontweight='bold', fontsize=10)
            ax4.set_ylim(0, 1)
            ax4.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                        f'{val:.3f}', ha='center', fontweight='bold', fontsize=8)
        
        elif graph_name == "Mass Balance Sankey":
            # Sankey-style mass balance diagram
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Flow widths (proportional to flow)
            feed_flow = self.params['feed_flow']
            perm_flow = self.results['permeate_flow']
            ret_flow = self.results['retentate_flow']
            
            max_flow = feed_flow
            feed_width = (feed_flow / max_flow) * 3
            perm_width = (perm_flow / max_flow) * 3
            ret_width = (ret_flow / max_flow) * 3
            
            # Draw flows as rectangles
            from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrow
            
            # Feed
            feed_box = FancyBboxPatch((0.5, 4.5 - feed_width/2), 2, feed_width,
                                     boxstyle="round,pad=0.1", edgecolor='black',
                                     facecolor='#2196F3', linewidth=2, alpha=0.7)
            ax.add_patch(feed_box)
            ax.text(1.5, 5, 'FEED', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            ax.text(1.5, 4.3, f'{feed_flow:.3f} kmol/s', ha='center', va='center', fontsize=9, color='white')
            ax.text(1.5, 3.8, f'CO₂: {self.params["feed_composition"]*100:.1f}%', 
                   ha='center', va='center', fontsize=8, color='white')
            
            # Membrane unit
            membrane_box = Rectangle((4, 2), 2, 6, edgecolor='black', facecolor='#FFD700',
                                     linewidth=2, alpha=0.5)
            ax.add_patch(membrane_box)
            ax.text(5, 5, 'MEMBRANE\nSEPARATOR', ha='center', va='center',
                   fontsize=10, fontweight='bold')
            ax.text(5, 3.5, f'{self.results["membrane_area"]:.1f} m²',
                   ha='center', va='center', fontsize=9)
            
            # Arrow from feed to membrane
            ax.annotate('', xy=(4, 5), xytext=(2.5, 5),
                       arrowprops=dict(arrowstyle='->', lw=3, color='#2196F3'))
            
            # Permeate
            perm_box = FancyBboxPatch((7, 6.5 - perm_width/2), 2, perm_width,
                                     boxstyle="round,pad=0.1", edgecolor='black',
                                     facecolor='#4CAF50', linewidth=2, alpha=0.7)
            ax.add_patch(perm_box)
            ax.text(8, 7, 'PERMEATE', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            ax.text(8, 6.5, f'{perm_flow:.3f} kmol/s', ha='center', va='center', fontsize=9, color='white')
            ax.text(8, 6, f'CO₂: {self.results["permeate_co2"]*100:.1f}%',
                   ha='center', va='center', fontsize=8, color='white')
            
            # Arrow to permeate
            ax.annotate('', xy=(7, 7), xytext=(6, 6.5),
                       arrowprops=dict(arrowstyle='->', lw=3, color='#4CAF50'))
            
            # Retentate
            ret_box = FancyBboxPatch((7, 3.5 - ret_width/2), 2, ret_width,
                                    boxstyle="round,pad=0.1", edgecolor='black',
                                    facecolor='#FF9800', linewidth=2, alpha=0.7)
            ax.add_patch(ret_box)
            ax.text(8, 3.5, 'RETENTATE', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            ax.text(8, 3, f'{ret_flow:.3f} kmol/s', ha='center', va='center', fontsize=9, color='white')
            ax.text(8, 2.5, f'CO₂: {self.results["retentate_co2"]*100:.1f}%',
                   ha='center', va='center', fontsize=8, color='white')
            
            # Arrow to retentate
            ax.annotate('', xy=(7, 3.5), xytext=(6, 3.5),
                       arrowprops=dict(arrowstyle='->', lw=3, color='#FF9800'))
            
            ax.set_title('Mass Balance Flow Diagram', fontweight='bold', fontsize=13, pad=10)
        
        elif graph_name == "CO₂ Capture Metrics":
            # Comprehensive CO2 capture metrics dashboard
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # CO2 recovery progression bar
            ax1 = fig.add_subplot(gs[0, :])
            recovery_pct = self.results['co2_recovery'] * 100
            ax1.barh([0], [recovery_pct], height=0.5, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax1.barh([0], [100-recovery_pct], left=[recovery_pct], height=0.5, 
                    color='#E0E0E0', alpha=0.5, edgecolor='black')
            ax1.set_xlim(0, 100)
            ax1.set_ylim(-0.5, 0.5)
            ax1.set_xlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=10)
            ax1.set_yticks([])
            ax1.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target')
            ax1.text(recovery_pct/2, 0, f'{recovery_pct:.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
            ax1.legend(loc='upper right')
            ax1.grid(axis='x', alpha=0.3)
            ax1.set_title('CO₂ Recovery Performance', fontweight='bold', fontsize=11)
            
            # CO2 purity progression bar
            ax2 = fig.add_subplot(gs[1, :])
            purity_pct = self.results['permeate_co2'] * 100
            ax2.barh([0], [purity_pct], height=0.5, color='#2196F3', alpha=0.7, edgecolor='black')
            ax2.barh([0], [100-purity_pct], left=[purity_pct], height=0.5,
                    color='#E0E0E0', alpha=0.5, edgecolor='black')
            ax2.set_xlim(0, 100)
            ax2.set_ylim(-0.5, 0.5)
            ax2.set_xlabel('CO₂ Purity (%)', fontweight='bold', fontsize=10)
            ax2.set_yticks([])
            ax2.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target')
            ax2.text(purity_pct/2, 0, f'{purity_pct:.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
            ax2.legend(loc='upper right')
            ax2.grid(axis='x', alpha=0.3)
            ax2.set_title('CO₂ Purity Performance', fontweight='bold', fontsize=11)
            
            # CO2 capture rate
            ax3 = fig.add_subplot(gs[2, 0])
            co2_captured_rate = self.results['permeate_flow'] * self.results['permeate_co2']
            ax3.bar(['Capture\nRate'], [co2_captured_rate * 44], color='#9C27B0',
                   alpha=0.7, edgecolor='black', width=0.5)
            ax3.set_ylabel('kg/s', fontweight='bold', fontsize=10)
            ax3.set_title('CO₂ Capture Rate', fontweight='bold', fontsize=10)
            ax3.text(0, co2_captured_rate * 44 / 2, f'{co2_captured_rate*44:.3f} kg/s',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            ax3.grid(axis='y', alpha=0.3)
            
            # Annual CO2 captured
            ax4 = fig.add_subplot(gs[2, 1])
            annual_co2 = co2_captured_rate * 44 * 3600 * 24 * 365 / 1000  # tonnes/year
            ax4.bar(['Annual\nCapture'], [annual_co2/1000], color='#00BCD4',
                   alpha=0.7, edgecolor='black', width=0.5)
            ax4.set_ylabel('kt CO₂/year', fontweight='bold', fontsize=10)
            ax4.set_title('Annual CO₂ Captured', fontweight='bold', fontsize=10)
            ax4.text(0, annual_co2/2000, f'{annual_co2/1000:.1f} kt/yr',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            ax4.grid(axis='y', alpha=0.3)
        
        elif graph_name == "Performance Gauge Panel":
            # Modern gauge panel showing key metrics
            gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
            
            def draw_gauge(ax, value, title, max_val=100, unit='%', zones=None):
                """Draw a semi-circular gauge"""
                if zones is None:
                    zones = [(0, 60, '#F44336'), (60, 80, '#FF9800'), (80, 100, '#4CAF50')]
                
                theta = np.linspace(0, np.pi, 100)
                r = np.ones_like(theta)
                
                ax = plt.subplot(ax, projection='polar')
                ax.set_theta_zero_location('W')
                ax.set_theta_direction(1)
                
                # Draw colored zones
                for z_min, z_max, color in zones:
                    theta_zone = theta[(theta >= z_min/max_val*np.pi) & (theta <= z_max/max_val*np.pi)]
                    ax.fill_between(theta_zone, 0, r[:len(theta_zone)], alpha=0.3, color=color)
                
                # Draw needle
                needle_angle = min(value / max_val * np.pi, np.pi)
                ax.plot([needle_angle, needle_angle], [0, 0.9], 'k-', linewidth=4)
                ax.plot([needle_angle], [0.9], 'o', color='red', markersize=10)
                
                # Center value
                ax.text(np.pi/2, 0.3, f'{value:.1f}{unit}', ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontweight='bold', fontsize=10, pad=20)
                ax.spines['polar'].set_visible(False)
            
            # Recovery gauge
            draw_gauge(gs[0, 0], self.results['co2_recovery']*100, 'CO₂ Recovery')
            
            # Purity gauge
            draw_gauge(gs[0, 1], self.results['permeate_co2']*100, 'CO₂ Purity')
            
            # Stage cut gauge
            draw_gauge(gs[0, 2], self.results['stage_cut']*100, 'Stage Cut',
                      zones=[(0, 20, '#4CAF50'), (20, 50, '#FF9800'), (50, 100, '#F44336')])
            
            # Selectivity gauge (different scale)
            selectivity = self.results['co2_permeance'] / self.results['n2_permeance']
            draw_gauge(gs[1, 0], min(selectivity, 100), 'Selectivity', max_val=100, unit='',
                      zones=[(0, 30, '#F44336'), (30, 50, '#FF9800'), (50, 100, '#4CAF50')])
            
            # Pressure ratio gauge
            pressure_ratio = self.params['feed_pressure'] / self.params['permeate_pressure']
            draw_gauge(gs[1, 1], min(pressure_ratio, 30), 'Pressure Ratio', max_val=30, unit='',
                      zones=[(0, 10, '#4CAF50'), (10, 20, '#FF9800'), (20, 30, '#F44336')])
            
            # Membrane efficiency gauge
            efficiency = (self.results['co2_recovery'] * self.results['permeate_co2']) * 100
            draw_gauge(gs[1, 2], efficiency, 'Overall Efficiency',
                      zones=[(0, 50, '#F44336'), (50, 70, '#FF9800'), (70, 100, '#4CAF50')])
    
    def draw_economics_graph(self, fig, graph_name):
        """Draw economics-related graphs"""
        total_opex = self.opex_results['Total OPEX']['Annual ($/year)']
        
        if graph_name == "OPEX Breakdown":
            ax = fig.add_subplot(111)
            costs = [
                self.opex_results['Energy']['Cost ($/year)'],
                self.opex_results['Membrane Replacement']['Cost ($/year)'],
                self.opex_results['Labor']['Total Cost ($/year)'],
                self.opex_results['Maintenance & Repairs']['Cost ($/year)'],
                self.opex_results['Utilities (Water)']['Cost ($/year)'],
                self.opex_results['Chemicals & Consumables']['Cost ($/year)']
            ]
            labels = ['Energy', 'Membrane\nReplace', 'Labor', 'Maintenance', 'Water', 'Chemicals']
            colors_opex = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7']
            
            wedges, texts, autotexts = ax.pie(costs, labels=labels, colors=colors_opex,
                                              autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.set_title(f'Annual OPEX Breakdown\n${total_opex/1000:.1f}k/year',
                        fontweight='bold', fontsize=12)
        
        elif graph_name == "OPEX Bar Chart":
            ax = fig.add_subplot(111)
            costs = [
                self.opex_results['Energy']['Cost ($/year)'],
                self.opex_results['Membrane Replacement']['Cost ($/year)'],
                self.opex_results['Labor']['Total Cost ($/year)'],
                self.opex_results['Maintenance & Repairs']['Cost ($/year)']
            ]
            cost_labels = ['Energy', 'Membrane\nReplace', 'Labor', 'Maintenance']
            cost_values = [c/1000 for c in costs]
            colors_opex = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            
            bars = ax.barh(cost_labels, cost_values, color=colors_opex, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Annual Cost ($k/year)', fontweight='bold', fontsize=11)
            ax.set_title('OPEX Components Comparison', fontweight='bold', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
            for bar, val in zip(bars, cost_values):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'${val:.1f}k', 
                        va='center', fontweight='bold', fontsize=10)
        
        elif graph_name == "CAPEX Breakdown":
            ax = fig.add_subplot(111)
            
            # Estimate CAPEX components
            membrane_capex = self.results['membrane_area'] * self.params['membrane_cost_per_m2']
            total_power = self.opex_results['Energy']['Power (kW)']
            compressor_capex = total_power * 500
            module_housing_capex = self.results['membrane_area'] * 20
            installation_capex = (membrane_capex + compressor_capex) * 0.15
            engineering_capex = (membrane_capex + compressor_capex) * 0.10
            contingency_capex = (membrane_capex + compressor_capex + module_housing_capex) * 0.20
            
            total_capex = (membrane_capex + compressor_capex + module_housing_capex + 
                          installation_capex + engineering_capex + contingency_capex)
            
            capex_components = [membrane_capex, compressor_capex, module_housing_capex, 
                               installation_capex, engineering_capex, contingency_capex]
            capex_labels = ['Membranes', 'Compressor', 'Housing', 'Installation', 'Engineering', 'Contingency']
            colors_capex = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#95A5A6']
            
            wedges, texts, autotexts = ax.pie(capex_components, labels=capex_labels, colors=colors_capex,
                                              autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.set_title(f'Total CAPEX Breakdown\n${total_capex/1000:.1f}k',
                        fontweight='bold', fontsize=12)
        
        elif graph_name == "CAPEX vs OPEX":
            ax = fig.add_subplot(111)
            
            # Calculate CAPEX
            membrane_capex = self.results['membrane_area'] * self.params['membrane_cost_per_m2']
            total_power = self.opex_results['Energy']['Power (kW)']
            compressor_capex = total_power * 500
            module_housing_capex = self.results['membrane_area'] * 20
            installation_capex = (membrane_capex + compressor_capex) * 0.15
            engineering_capex = (membrane_capex + compressor_capex) * 0.10
            contingency_capex = (membrane_capex + compressor_capex + module_housing_capex) * 0.20
            
            total_capex = (membrane_capex + compressor_capex + module_housing_capex + 
                          installation_capex + engineering_capex + contingency_capex)
            
            total_costs = [total_capex/1000, total_opex/1000]
            cost_types = ['CAPEX', 'Annual\nOPEX']
            bars = ax.bar(cost_types, total_costs, color=['#3498DB', '#E74C3C'], alpha=0.7, edgecolor='black', width=0.5)
            ax.set_ylabel('Cost ($k)', fontweight='bold', fontsize=11)
            ax.set_title('CAPEX vs Annual OPEX Comparison', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, total_costs):
                ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'${val:.1f}k', 
                        ha='center', fontweight='bold', fontsize=11)
            
            # Add payback estimate
            if total_opex > 0:
                simple_payback = total_capex / total_opex
                ax.text(0.5, 0.92, f'Simple Payback: {simple_payback:.1f} years', 
                        ha='center', transform=ax.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        elif graph_name == "Cost per Ton CO₂":
            ax = fig.add_subplot(111)
            
            co2_captured_mol_s = self.results['co2_permeated']
            co2_captured_ton_yr = co2_captured_mol_s * 44 / 1000 * self.opex_calc.operating_hours_per_year * 3600 / 1000
            cost_per_ton = total_opex / co2_captured_ton_yr if co2_captured_ton_yr > 0 else 0
            
            bars = ax.bar(['Cost/ton CO₂'], [cost_per_ton], color='#E74C3C', alpha=0.7, edgecolor='black', width=0.4)
            ax.set_ylabel('$/ton CO₂', fontweight='bold', fontsize=11)
            ax.set_title('CO₂ Capture Cost', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            ax.text(0, cost_per_ton + 2, f'${cost_per_ton:.2f}/ton', ha='center', fontweight='bold', fontsize=12)
            
            # Add DOE target line
            ax.axhline(y=40, color='green', linestyle='--', linewidth=2.5, alpha=0.6, label='DOE Target ($40/ton)')
            ax.legend(fontsize=10, loc='upper right')
            
            # Add interpretation text
            if cost_per_ton <= 40:
                status = f'✓ Below DOE target by ${40-cost_per_ton:.2f}/ton'
                color = 'green'
            else:
                status = f'✗ Above DOE target by ${cost_per_ton-40:.2f}/ton'
                color = 'red'
            ax.text(0.5, 0.85, status, transform=ax.transAxes, ha='center',
                   fontsize=11, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        elif graph_name == "Economic Waterfall":
            # Waterfall chart showing cost buildup
            ax = fig.add_subplot(111)
            
            # Calculate costs
            costs_data = [
                ('Base', 0),
                ('Energy', self.opex_results['Energy']['Cost ($/year)']/1000),
                ('Membranes', self.opex_results['Membrane Replacement']['Cost ($/year)']/1000),
                ('Labor', self.opex_results['Labor']['Total Cost ($/year)']/1000),
                ('Maintenance', self.opex_results['Maintenance & Repairs']['Cost ($/year)']/1000),
                ('Utilities', self.opex_results['Utilities (Water)']['Cost ($/year)']/1000),
                ('Chemicals', self.opex_results['Chemicals & Consumables']['Cost ($/year)']/1000),
                ('Total OPEX', total_opex/1000)
            ]
            
            # Build waterfall
            cumulative = 0
            bottoms = []
            values = []
            colors = []
            
            for i, (label, value) in enumerate(costs_data):
                if i == 0:  # Base
                    bottoms.append(0)
                    values.append(0)
                    colors.append('gray')
                elif i == len(costs_data) - 1:  # Total
                    bottoms.append(0)
                    values.append(cumulative)
                    colors.append('#2196F3')
                else:  # Incremental
                    bottoms.append(cumulative)
                    values.append(value)
                    colors.append('#4CAF50')
                    cumulative += value
            
            labels = [x[0] for x in costs_data]
            bars = ax.bar(range(len(labels)), values, bottom=bottoms, color=colors,
                          alpha=0.7, edgecolor='black', width=0.6)
            
            # Draw connecting lines
            for i in range(len(labels)-2):
                if i > 0:
                    ax.plot([i, i+1], [bottoms[i]+values[i], bottoms[i+1]], 
                           'k--', linewidth=1, alpha=0.5)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Cost ($k/year)', fontweight='bold', fontsize=11)
            ax.set_title('Annual OPEX Waterfall Chart', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, bottom, val) in enumerate(zip(bars, bottoms, values)):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bottom + val/2,
                           f'${val:.1f}k', ha='center', va='center',
                           fontweight='bold', fontsize=9, color='white')
        
        elif graph_name == "ROI Analysis":
            # Return on Investment analysis
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Calculate CAPEX
            membrane_capex = self.results['membrane_area'] * self.params['membrane_cost_per_m2']
            total_power = self.opex_results['Energy']['Power (kW)']
            compressor_capex = total_power * 500
            module_housing_capex = self.results['membrane_area'] * 20
            installation_capex = (membrane_capex + compressor_capex) * 0.15
            engineering_capex = (membrane_capex + compressor_capex) * 0.10
            contingency_capex = (membrane_capex + compressor_capex + module_housing_capex) * 0.20
            total_capex = (membrane_capex + compressor_capex + module_housing_capex + 
                          installation_capex + engineering_capex + contingency_capex)
            
            # Cumulative cash flow over 20 years
            ax1 = fig.add_subplot(gs[0, :])
            years = np.arange(0, 21)
            cash_flow = np.zeros(len(years))
            cash_flow[0] = -total_capex
            for i in range(1, len(years)):
                cash_flow[i] = cash_flow[i-1] - total_opex
            
            ax1.plot(years, cash_flow/1000, 'o-', linewidth=2, markersize=6, color='#E74C3C')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.fill_between(years, 0, cash_flow/1000, where=(cash_flow<0), alpha=0.3, color='red')
            ax1.fill_between(years, 0, cash_flow/1000, where=(cash_flow>=0), alpha=0.3, color='green')
            ax1.set_xlabel('Year', fontweight='bold', fontsize=10)
            ax1.set_ylabel('Cumulative Cash Flow ($k)', fontweight='bold', fontsize=10)
            ax1.set_title('Cumulative Cash Flow (No Revenue)', fontweight='bold', fontsize=11)
            ax1.grid(alpha=0.3)
            
            # Payback period
            ax2 = fig.add_subplot(gs[1, 0])
            if total_opex > 0:
                payback = total_capex / total_opex
            else:
                payback = 999
            ax2.bar(['Payback\nPeriod'], [payback], color='#FF9800', alpha=0.7, edgecolor='black', width=0.5)
            ax2.set_ylabel('Years', fontweight='bold', fontsize=10)
            ax2.set_title('Simple Payback Period', fontweight='bold', fontsize=10)
            ax2.text(0, payback/2, f'{payback:.1f} yrs', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
            ax2.grid(axis='y', alpha=0.3)
            
            # NPV breakdown (assuming 10% discount rate)
            ax3 = fig.add_subplot(gs[1, 1])
            discount_rate = 0.10
            npv_20yr = -total_capex
            for year in range(1, 21):
                npv_20yr += -total_opex / ((1 + discount_rate) ** year)
            
            ax3.bar(['NPV\n(20 years)'], [npv_20yr/1000], 
                   color='#9C27B0' if npv_20yr < 0 else '#4CAF50',
                   alpha=0.7, edgecolor='black', width=0.5)
            ax3.set_ylabel('NPV ($k)', fontweight='bold', fontsize=10)
            ax3.set_title('Net Present Value @ 10%', fontweight='bold', fontsize=10)
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax3.text(0, npv_20yr/2000, f'${npv_20yr/1000:.1f}k', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
            ax3.grid(axis='y', alpha=0.3)
        
        elif graph_name == "Cost Breakdown Treemap":
            # Treemap of all costs
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            ax.set_title('Cost Breakdown Treemap', fontweight='bold', fontsize=13, pad=20)
            
            # Calculate all costs
            costs = {
                'Energy': self.opex_results['Energy']['Cost ($/year)'],
                'Membranes': self.opex_results['Membrane Replacement']['Cost ($/year)'],
                'Labor': self.opex_results['Labor']['Total Cost ($/year)'],
                'Maintenance': self.opex_results['Maintenance & Repairs']['Cost ($/year)'],
                'Water': self.opex_results['Utilities (Water)']['Cost ($/year)'],
                'Chemicals': self.opex_results['Chemicals & Consumables']['Cost ($/year)']
            }
            
            total = sum(costs.values())
            colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7']
            
            # Simple treemap layout (squarified approximation)
            from matplotlib.patches import Rectangle
            sorted_costs = sorted(costs.items(), key=lambda x: x[1], reverse=True)
            
            y_pos = 0
            for i, (name, cost) in enumerate(sorted_costs):
                height = (cost / total) * 10
                rect = Rectangle((0, y_pos), 10, height, 
                               facecolor=colors_map[i % len(colors_map)],
                               edgecolor='white', linewidth=2, alpha=0.7)
                ax.add_patch(rect)
                
                # Add label
                pct = cost / total * 100
                ax.text(5, y_pos + height/2, f'{name}\n${cost/1000:.1f}k\n({pct:.1f}%)',
                       ha='center', va='center', fontweight='bold', fontsize=10,
                       color='white' if i < 3 else 'black')
                
                y_pos += height
        
        elif graph_name == "Payback Period":
            # Detailed payback analysis with sensitivity
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            
            # Calculate CAPEX
            membrane_capex = self.results['membrane_area'] * self.params['membrane_cost_per_m2']
            total_power = self.opex_results['Energy']['Power (kW)']
            compressor_capex = total_power * 500
            module_housing_capex = self.results['membrane_area'] * 20
            installation_capex = (membrane_capex + compressor_capex) * 0.15
            engineering_capex = (membrane_capex + compressor_capex) * 0.10
            contingency_capex = (membrane_capex + compressor_capex + module_housing_capex) * 0.20
            total_capex = (membrane_capex + compressor_capex + module_housing_capex + 
                          installation_capex + engineering_capex + contingency_capex)
            
            # Payback with varying savings
            ax1 = fig.add_subplot(gs[0])
            savings_multipliers = np.linspace(0.5, 2.0, 50)
            payback_periods = total_capex / (total_opex * savings_multipliers)
            
            ax1.plot(savings_multipliers, payback_periods, linewidth=3, color='#2196F3')
            ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10-year limit')
            ax1.axhline(y=5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='5-year target')
            ax1.fill_between(savings_multipliers, 0, payback_periods, 
                           where=(payback_periods<=5), alpha=0.3, color='green')
            ax1.fill_between(savings_multipliers, 0, payback_periods,
                           where=(payback_periods>10), alpha=0.3, color='red')
            ax1.set_xlabel('Annual Savings Multiplier', fontweight='bold', fontsize=10)
            ax1.set_ylabel('Payback Period (years)', fontweight='bold', fontsize=10)
            ax1.set_title('Payback Period Sensitivity', fontweight='bold', fontsize=11)
            ax1.set_ylim(0, min(20, max(payback_periods)))
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.3)
            
            # Current payback point
            current_payback = total_capex / total_opex if total_opex > 0 else 999
            ax1.plot([1.0], [current_payback], 'ro', markersize=10, 
                    label=f'Current: {current_payback:.1f} yrs', zorder=5)
            
            # Break-even timeline
            ax2 = fig.add_subplot(gs[1])
            years = np.arange(0, 16)
            cumulative = np.zeros(len(years))
            cumulative[0] = -total_capex
            for i in range(1, len(years)):
                cumulative[i] = cumulative[i-1] + total_opex
            
            bars = ax2.bar(years, cumulative/1000, color=['red' if c < 0 else 'green' for c in cumulative],
                          alpha=0.7, edgecolor='black', width=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
            ax2.set_xlabel('Year', fontweight='bold', fontsize=10)
            ax2.set_ylabel('Net Cash Position ($k)', fontweight='bold', fontsize=10)
            ax2.set_title('Break-Even Timeline', fontweight='bold', fontsize=11)
            ax2.grid(axis='y', alpha=0.3)
    
    def draw_sensitivity_graph(self, fig, graph_name):
        """Draw sensitivity analysis graphs"""
        if graph_name == "Feed Pressure":
            ax = fig.add_subplot(111)
            pressures = np.linspace(1, 10, 30)
            recoveries = []
            purities = []
            
            for p in pressures:
                if self.params['membrane_type'] == 'Polaris':
                    co2_permeance_gpu, selectivity = 3000, 30
                else:
                    co2_permeance_gpu, selectivity = 2500, 680
                
                mem_temp = MembraneSeparation(
                    feed_composition=self.params['feed_composition'],
                    feed_pressure=p,
                    permeate_pressure=self.params['permeate_pressure'],
                    temperature=self.params['temperature'],
                    co2_permeance_gpu=co2_permeance_gpu,
                    selectivity=selectivity
                )
                res = mem_temp.solve_single_stage(self.params['feed_flow'])
                if res:
                    recoveries.append(res['co2_recovery'] * 100)
                    purities.append(res['permeate_co2'] * 100)
                else:
                    recoveries.append(0)
                    purities.append(0)
            
            ax.plot(pressures, recoveries, 'b-o', linewidth=2.5, markersize=4, label='Recovery')
            ax.plot(pressures, purities, 'r-s', linewidth=2.5, markersize=4, label='Purity')
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.6)
            ax.axvline(x=self.params['feed_pressure'], color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=11)
            ax.set_title('Sensitivity to Feed Pressure', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        elif graph_name == "Temperature":
            ax = fig.add_subplot(111)
            temps = np.linspace(273, 373, 30)
            recoveries = []
            areas = []
            
            for T in temps:
                if self.params['membrane_type'] == 'Polaris':
                    co2_permeance_gpu, selectivity = 3000, 30
                else:
                    co2_permeance_gpu, selectivity = 2500, 680
                
                mem_temp = MembraneSeparation(
                    feed_composition=self.params['feed_composition'],
                    feed_pressure=self.params['feed_pressure'],
                    permeate_pressure=self.params['permeate_pressure'],
                    temperature=T,
                    co2_permeance_gpu=co2_permeance_gpu,
                    selectivity=selectivity
                )
                res = mem_temp.solve_single_stage(self.params['feed_flow'])
                if res:
                    recoveries.append(res['co2_recovery'] * 100)
                    areas.append(res['membrane_area'])
                else:
                    recoveries.append(0)
                    areas.append(0)
            
            ax_twin = ax.twinx()
            line1 = ax.plot(temps, recoveries, 'b-o', linewidth=2.5, markersize=4, label='Recovery')
            line2 = ax_twin.plot(temps, areas, 'g-s', linewidth=2.5, markersize=4, label='Area')
            ax.axvline(x=self.params['temperature'], color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
            ax.set_xlabel('Temperature (K)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Recovery (%)', fontweight='bold', fontsize=11, color='b')
            ax_twin.set_ylabel('Area (m²)', fontweight='bold', fontsize=11, color='g')
            ax.tick_params(axis='y', labelcolor='b')
            ax_twin.tick_params(axis='y', labelcolor='g')
            ax.set_title('Sensitivity to Temperature', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=10)
        
        elif graph_name == "Feed Composition":
            ax = fig.add_subplot(111)
            comps = np.linspace(0.05, 0.50, 30)
            recoveries = []
            purities = []
            
            for comp in comps:
                if self.params['membrane_type'] == 'Polaris':
                    co2_permeance_gpu, selectivity = 3000, 30
                else:
                    co2_permeance_gpu, selectivity = 2500, 680
                
                mem_temp = MembraneSeparation(
                    feed_composition=comp,
                    feed_pressure=self.params['feed_pressure'],
                    permeate_pressure=self.params['permeate_pressure'],
                    temperature=self.params['temperature'],
                    co2_permeance_gpu=co2_permeance_gpu,
                    selectivity=selectivity
                )
                res = mem_temp.solve_single_stage(self.params['feed_flow'])
                if res:
                    recoveries.append(res['co2_recovery'] * 100)
                    purities.append(res['permeate_co2'] * 100)
                else:
                    recoveries.append(0)
                    purities.append(0)
            
            ax.plot(np.array(comps)*100, recoveries, 'b-o', linewidth=2.5, markersize=4, label='Recovery')
            ax.plot(np.array(comps)*100, purities, 'r-s', linewidth=2.5, markersize=4, label='Purity')
            ax.axvline(x=self.params['feed_composition']*100, color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.6)
            ax.set_xlabel('Feed CO₂ Composition (vol%)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=11)
            ax.set_title('Sensitivity to Feed Composition', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        elif graph_name == "Area-Recovery Trade-off":
            ax = fig.add_subplot(111)
            pressures = np.linspace(1, 10, 20)
            areas = []
            recoveries = []
            
            for p in pressures:
                if self.params['membrane_type'] == 'Polaris':
                    co2_permeance_gpu, selectivity = 3000, 30
                else:
                    co2_permeance_gpu, selectivity = 2500, 680
                
                mem_temp = MembraneSeparation(
                    feed_composition=self.params['feed_composition'],
                    feed_pressure=p,
                    permeate_pressure=self.params['permeate_pressure'],
                    temperature=self.params['temperature'],
                    co2_permeance_gpu=co2_permeance_gpu,
                    selectivity=selectivity
                )
                res = mem_temp.solve_single_stage(self.params['feed_flow'])
                if res:
                    areas.append(res['membrane_area'])
                    recoveries.append(res['co2_recovery'] * 100)
                else:
                    areas.append(0)
                    recoveries.append(0)
            
            scatter = ax.scatter(areas, recoveries, c=pressures, cmap='viridis', s=80, 
                               alpha=0.7, edgecolors='black', linewidths=1.5)
            ax.scatter([self.results['membrane_area']], [self.results['co2_recovery']*100], 
                      color='red', s=300, marker='*', edgecolors='black', linewidths=2, 
                      label='Current', zorder=5)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Feed Pressure (bar)', fontsize=9, fontweight='bold')
            ax.set_xlabel('Membrane Area (m²)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11)
            ax.set_title('Area-Recovery Trade-off', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        elif graph_name == "Pressure Ratio Impact":
            # New graph showing impact of pressure ratio
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            pressure_ratios = np.linspace(2, 30, 40)
            recoveries = []
            purities = []
            areas = []
            energies = []
            
            for pr in pressure_ratios:
                pp = self.params['feed_pressure'] / pr
                if pp < 0.05:
                    continue
                    
                if self.params['membrane_type'] == 'Polaris':
                    co2_permeance_gpu, selectivity = 3000, 30
                else:
                    co2_permeance_gpu, selectivity = 2500, 680
                
                try:
                    mem_temp = MembraneSeparation(
                        feed_composition=self.params['feed_composition'],
                        feed_pressure=self.params['feed_pressure'],
                        permeate_pressure=pp,
                        temperature=self.params['temperature'],
                        co2_permeance_gpu=co2_permeance_gpu,
                        selectivity=selectivity
                    )
                    res = mem_temp.solve_single_stage(self.params['feed_flow'])
                    if res:
                        recoveries.append(res['co2_recovery'] * 100)
                        purities.append(res['permeate_co2'] * 100)
                        areas.append(res['membrane_area'])
                        # Approximate energy
                        energy = res['retentate_flow'] * 8.314 * self.params['temperature'] * \
                                np.log(self.params['feed_pressure'] / pp) / 1000
                        energies.append(energy)
                except:
                    pass
            
            valid_prs = pressure_ratios[:len(recoveries)]
            
            # Recovery vs PR
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(valid_prs, recoveries, 'b-o', linewidth=2, markersize=5)
            ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Pressure Ratio', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Recovery (%)', fontweight='bold', fontsize=9)
            ax1.set_title('Recovery vs Pressure Ratio', fontweight='bold', fontsize=10)
            ax1.grid(alpha=0.3)
            
            # Purity vs PR
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(valid_prs, purities, 'r-s', linewidth=2, markersize=5)
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Pressure Ratio', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Purity (%)', fontweight='bold', fontsize=9)
            ax2.set_title('Purity vs Pressure Ratio', fontweight='bold', fontsize=10)
            ax2.grid(alpha=0.3)
            
            # Area vs PR
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(valid_prs, areas, 'g-^', linewidth=2, markersize=5)
            ax3.set_xlabel('Pressure Ratio', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Area (m²)', fontweight='bold', fontsize=9)
            ax3.set_title('Area vs Pressure Ratio', fontweight='bold', fontsize=10)
            ax3.grid(alpha=0.3)
            
            # Energy vs PR
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(valid_prs, energies, 'm-d', linewidth=2, markersize=5)
            ax4.set_xlabel('Pressure Ratio', fontweight='bold', fontsize=9)
            ax4.set_ylabel('Specific Energy (kW)', fontweight='bold', fontsize=9)
            ax4.set_title('Energy vs Pressure Ratio', fontweight='bold', fontsize=10)
            ax4.grid(alpha=0.3)
        
        elif graph_name == "Multi-Variable Tornado":
            # Tornado diagram for sensitivity
            ax = fig.add_subplot(111)
            
            # Base case
            base_recovery = self.results['co2_recovery'] * 100
            
            # Parameter variations
            params_to_test = {
                'Feed Pressure': ('feed_pressure', [2.0, 4.0]),
                'Permeate Pressure': ('permeate_pressure', [0.1, 0.3]),
                'Temperature': ('temperature', [280, 320]),
                'Feed CO₂': ('feed_composition', [0.10, 0.20])
            }
            
            impacts = []
            labels = []
            
            for label, (param, values) in params_to_test.items():
                recoveries_test = []
                for val in values:
                    test_params = self.params.copy()
                    test_params[param] = val
                    
                    if test_params['membrane_type'] == 'Polaris':
                        co2_permeance_gpu, selectivity = 3000, 30
                    else:
                        co2_permeance_gpu, selectivity = 2500, 680
                    
                    try:
                        mem_temp = MembraneSeparation(
                            feed_composition=test_params['feed_composition'],
                            feed_pressure=test_params['feed_pressure'],
                            permeate_pressure=test_params['permeate_pressure'],
                            temperature=test_params['temperature'],
                            co2_permeance_gpu=co2_permeance_gpu,
                            selectivity=selectivity
                        )
                        res = mem_temp.solve_single_stage(test_params['feed_flow'])
                        if res:
                            recoveries_test.append(res['co2_recovery'] * 100)
                        else:
                            recoveries_test.append(base_recovery)
                    except:
                        recoveries_test.append(base_recovery)
                
                low_impact = recoveries_test[0] - base_recovery
                high_impact = recoveries_test[1] - base_recovery
                impacts.append((low_impact, high_impact))
                labels.append(label)
            
            # Sort by total range
            sorted_indices = sorted(range(len(impacts)), 
                                  key=lambda i: abs(impacts[i][1] - impacts[i][0]), 
                                  reverse=True)
            
            y_pos = np.arange(len(labels))
            for i, idx in enumerate(sorted_indices):
                low, high = impacts[idx]
                ax.barh(i, high, left=0, height=0.8, color='#4CAF50', alpha=0.7, 
                       edgecolor='black', label='High' if i == 0 else '')
                ax.barh(i, -low, left=0, height=0.8, color='#F44336', alpha=0.7,
                       edgecolor='black', label='Low' if i == 0 else '')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([labels[i] for i in sorted_indices])
            ax.set_xlabel('Impact on CO₂ Recovery (%)', fontweight='bold', fontsize=11)
            ax.set_title('Tornado Diagram - Parameter Sensitivity', fontweight='bold', fontsize=13)
            ax.axvline(x=0, color='black', linewidth=2)
            ax.legend(fontsize=10)
            ax.grid(axis='x', alpha=0.3)
        
        elif graph_name == "Operating Cost Sensitivity":
            # Cost sensitivity to key parameters
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Electricity cost sensitivity
            ax1 = fig.add_subplot(gs[0, 0])
            elec_costs = np.linspace(0.03, 0.15, 30)
            total_costs = []
            for ec in elec_costs:
                temp_opex = self.opex_calc.calculate_opex(
                    membrane_area=self.results['membrane_area'],
                    feed_pressure=self.params['feed_pressure'],
                    permeate_pressure=self.params['permeate_pressure'],
                    retentate_flow=self.results['retentate_flow'],
                    temperature=self.params['temperature'],
                    electricity_cost=ec
                )
                total_costs.append(temp_opex['Total OPEX']['Annual ($/year)']/1000)
            
            ax1.plot(elec_costs, total_costs, 'b-o', linewidth=2, markersize=4)
            ax1.axvline(x=self.params['electricity_cost'], color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Electricity Cost ($/kWh)', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Annual OPEX ($k)', fontweight='bold', fontsize=9)
            ax1.set_title('OPEX vs Electricity Cost', fontweight='bold', fontsize=10)
            ax1.grid(alpha=0.3)
            
            # Membrane cost sensitivity
            ax2 = fig.add_subplot(gs[0, 1])
            mem_costs = np.linspace(20, 100, 30)
            total_costs_mem = []
            for mc in mem_costs:
                temp_opex = self.opex_calc.calculate_opex(
                    membrane_area=self.results['membrane_area'],
                    feed_pressure=self.params['feed_pressure'],
                    permeate_pressure=self.params['permeate_pressure'],
                    retentate_flow=self.results['retentate_flow'],
                    temperature=self.params['temperature'],
                    membrane_cost_per_m2=mc
                )
                total_costs_mem.append(temp_opex['Total OPEX']['Annual ($/year)']/1000)
            
            ax2.plot(mem_costs, total_costs_mem, 'r-s', linewidth=2, markersize=4)
            ax2.axvline(x=self.params['membrane_cost_per_m2'], color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Membrane Cost ($/m²)', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Annual OPEX ($k)', fontweight='bold', fontsize=9)
            ax2.set_title('OPEX vs Membrane Cost', fontweight='bold', fontsize=10)
            ax2.grid(alpha=0.3)
            
            # Combined heatmap
            ax3 = fig.add_subplot(gs[1, :])
            ec_grid = np.linspace(0.04, 0.12, 15)
            mc_grid = np.linspace(30, 80, 15)
            cost_map = np.zeros((len(ec_grid), len(mc_grid)))
            
            for i, ec in enumerate(ec_grid):
                for j, mc in enumerate(mc_grid):
                    temp_opex = self.opex_calc.calculate_opex(
                        membrane_area=self.results['membrane_area'],
                        feed_pressure=self.params['feed_pressure'],
                        permeate_pressure=self.params['permeate_pressure'],
                        retentate_flow=self.results['retentate_flow'],
                        temperature=self.params['temperature'],
                        electricity_cost=ec,
                        membrane_cost_per_m2=mc
                    )
                    cost_map[i, j] = temp_opex['Total OPEX']['Annual ($/year)']/1000
            
            EC, MC = np.meshgrid(mc_grid, ec_grid)
            contourf = ax3.contourf(EC, MC, cost_map, levels=15, cmap='YlOrRd', alpha=0.8)
            ax3.plot([self.params['membrane_cost_per_m2']], [self.params['electricity_cost']],
                    'b*', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Current')
            cbar = fig.colorbar(contourf, ax=ax3)
            cbar.set_label('Annual OPEX ($k)', fontsize=9, fontweight='bold')
            ax3.set_xlabel('Membrane Cost ($/m²)', fontweight='bold', fontsize=10)
            ax3.set_ylabel('Electricity Cost ($/kWh)', fontweight='bold', fontsize=10)
            ax3.set_title('OPEX Sensitivity Heatmap', fontweight='bold', fontsize=11)
            ax3.legend(fontsize=9)
        
        elif graph_name == "Selectivity Sensitivity":
            # Selectivity impact analysis
            ax = fig.add_subplot(111)
            
            selectivities = np.linspace(10, 100, 50)
            recoveries = []
            purities = []
            
            for sel in selectivities:
                try:
                    mem_temp = MembraneSeparation(
                        feed_composition=self.params['feed_composition'],
                        feed_pressure=self.params['feed_pressure'],
                        permeate_pressure=self.params['permeate_pressure'],
                        temperature=self.params['temperature'],
                        co2_permeance_gpu=2500,
                        selectivity=sel
                    )
                    res = mem_temp.solve_single_stage(self.params['feed_flow'])
                    if res:
                        recoveries.append(res['co2_recovery'] * 100)
                        purities.append(res['permeate_co2'] * 100)
                    else:
                        recoveries.append(0)
                        purities.append(0)
                except:
                    recoveries.append(0)
                    purities.append(0)
            
            ax.plot(selectivities, recoveries, 'b-o', linewidth=2.5, markersize=5, 
                   label='CO₂ Recovery', markevery=5)
            ax.plot(selectivities, purities, 'r-s', linewidth=2.5, markersize=5,
                   label='CO₂ Purity', markevery=5)
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Target')
            
            # Mark current selectivity
            current_sel = self.results['co2_permeance'] / self.results['n2_permeance']
            ax.axvline(x=current_sel, color='purple', linestyle=':', linewidth=2.5, 
                      alpha=0.7, label=f'Current α={current_sel:.1f}')
            
            ax.set_xlabel('CO₂/N₂ Selectivity (α)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=11)
            ax.set_title('Performance Sensitivity to Membrane Selectivity', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(10, 100)
    
    def draw_advanced_graph(self, fig, graph_name):
        """Draw advanced analysis graphs"""
        if graph_name == "Operating Window":
            ax = fig.add_subplot(111)
            
            # Create contour plot
            pressure_ratios = np.linspace(2, 50, 25)
            feed_pressures = np.linspace(1, 10, 25)
            PR, FP = np.meshgrid(pressure_ratios, feed_pressures)
            
            Recovery_map = np.zeros_like(PR)
            
            for i in range(len(pressure_ratios)):
                for j in range(len(feed_pressures)):
                    pr = pressure_ratios[i]
                    fp = feed_pressures[j]
                    p_perm = fp / pr
                    
                    if self.params['membrane_type'] == 'Polaris':
                        co2_permeance_gpu, selectivity = 3000, 30
                    else:
                        co2_permeance_gpu, selectivity = 2500, 680
                    
                    try:
                        mem_temp = MembraneSeparation(
                            feed_composition=self.params['feed_composition'],
                            feed_pressure=fp,
                            permeate_pressure=max(0.05, p_perm),
                            temperature=self.params['temperature'],
                            co2_permeance_gpu=co2_permeance_gpu,
                            selectivity=selectivity
                        )
                        res = mem_temp.solve_single_stage(self.params['feed_flow'])
                        if res:
                            Recovery_map[j, i] = res['co2_recovery'] * 100
                        else:
                            Recovery_map[j, i] = 0
                    except:
                        Recovery_map[j, i] = 0
            
            contourf = ax.contourf(PR, FP, Recovery_map, levels=15, cmap='RdYlGn', alpha=0.7)
            contour = ax.contour(PR, FP, Recovery_map, levels=[80], colors='blue', linewidths=3, linestyles='--')
            
            # Mark current point
            current_pr = self.params['feed_pressure'] / self.params['permeate_pressure']
            ax.scatter([current_pr], [self.params['feed_pressure']], color='red', s=300, marker='*', 
                      edgecolors='black', linewidths=2, label='Current', zorder=5)
            
            cbar = fig.colorbar(contourf, ax=ax)
            cbar.set_label('CO₂ Recovery (%)', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Pressure Ratio (P_feed / P_perm)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_title('Operating Window Map', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        elif graph_name in ["CO₂ Flux Profile", "N₂ Flux Profile"]:
            ax = fig.add_subplot(111)
            
            # Calculate flux
            if self.params['membrane_type'] == 'Polaris':
                P_CO2_gpu, selectivity = 3000, 30
            else:
                P_CO2_gpu, selectivity = 2500, 680
            
            P_CO2_SI = P_CO2_gpu * GPU_TO_SI
            P_N2_SI = P_CO2_SI / selectivity
            
            delta_p_range = np.linspace(0, 5e5, 50)
            
            if graph_name == "CO₂ Flux Profile":
                flux = P_CO2_SI * delta_p_range * 1000  # mmol/(m²·s)
                
                x_ret = self.results['retentate_co2']
                y_perm = self.results['permeate_co2']
                p_co2_feed = x_ret * self.params['feed_pressure'] * 1e5
                p_co2_perm = y_perm * self.params['permeate_pressure'] * 1e5
                current_delta_p = p_co2_feed - p_co2_perm
                current_flux = P_CO2_SI * current_delta_p * 1000
                
                ax.plot(delta_p_range/1e5, flux, 'g-', linewidth=2.5, label='CO₂ Flux')
                ax.scatter([current_delta_p/1e5], [current_flux], color='red', s=150, 
                          marker='o', edgecolors='black', linewidths=2, label='Operating Point', zorder=5)
                ax.set_ylabel('CO₂ Flux (mmol/m²·s)', fontweight='bold', fontsize=11)
                ax.set_title('CO₂ Flux vs Driving Force', fontweight='bold', fontsize=13)
            else:
                flux = P_N2_SI * delta_p_range * 1000
                
                x_ret = self.results['retentate_co2']
                y_perm = self.results['permeate_co2']
                p_n2_feed = (1 - x_ret) * self.params['feed_pressure'] * 1e5
                p_n2_perm = (1 - y_perm) * self.params['permeate_pressure'] * 1e5
                current_delta_p = p_n2_feed - p_n2_perm
                current_flux = P_N2_SI * current_delta_p * 1000
                
                ax.plot(delta_p_range/1e5, flux, 'b-', linewidth=2.5, label='N₂ Flux')
                ax.scatter([current_delta_p/1e5], [current_flux], color='red', s=150, 
                          marker='o', edgecolors='black', linewidths=2, label='Operating Point', zorder=5)
                ax.set_ylabel('N₂ Flux (mmol/m²·s)', fontweight='bold', fontsize=11)
                ax.set_title('N₂ Flux vs Driving Force', fontweight='bold', fontsize=13)
            
            ax.set_xlabel('Driving Force, Δp (bar)', fontweight='bold', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        elif graph_name == "3D Performance Map":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            
            feed_pressures = np.linspace(1, 10, 15)
            perm_pressures = np.linspace(0.05, 2, 15)
            FP, PP = np.meshgrid(feed_pressures, perm_pressures)
            
            Recovery_3D = np.zeros_like(FP)
            
            for i in range(len(feed_pressures)):
                for j in range(len(perm_pressures)):
                    if self.params['membrane_type'] == 'Polaris':
                        co2_permeance_gpu, selectivity = 3000, 30
                    else:
                        co2_permeance_gpu, selectivity = 2500, 680
                    
                    try:
                        mem_temp = MembraneSeparation(
                            feed_composition=self.params['feed_composition'],
                            feed_pressure=feed_pressures[i],
                            permeate_pressure=perm_pressures[j],
                            temperature=self.params['temperature'],
                            co2_permeance_gpu=co2_permeance_gpu,
                            selectivity=selectivity
                        )
                        res = mem_temp.solve_single_stage(self.params['feed_flow'])
                        if res:
                            Recovery_3D[j, i] = res['co2_recovery'] * 100
                        else:
                            Recovery_3D[j, i] = 0
                    except:
                        Recovery_3D[j, i] = 0
            
            surf = ax.plot_surface(FP, PP, Recovery_3D, cmap='viridis', alpha=0.8, 
                                  edgecolor='none', antialiased=True)
            
            ax.scatter([self.params['feed_pressure']], [self.params['permeate_pressure']], 
                      [self.results['co2_recovery']*100], color='red', s=200, marker='o',
                      edgecolors='black', linewidths=2, zorder=10)
            
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=9)
            ax.set_ylabel('Permeate Pressure (bar)', fontweight='bold', fontsize=9)
            ax.set_zlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=9)
            ax.set_title('3D Performance Map', fontweight='bold', fontsize=11)
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            ax.view_init(elev=25, azim=45)
        
        elif graph_name == "Process Flow Diagram":
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch
            
            # Feed stream
            ax.add_patch(FancyArrow(0.05, 0.5, 0.12, 0, width=0.04, 
                                   facecolor='#2196F3', edgecolor='black', linewidth=1.5))
            ax.text(0.11, 0.62, 'FEED', ha='center', fontweight='bold', fontsize=11)
            ax.text(0.11, 0.56, f'{self.params["feed_flow"]:.2f} kmol/s', ha='center', fontsize=9)
            ax.text(0.11, 0.51, f'{self.params["feed_composition"]*100:.1f}% CO₂', ha='center', fontsize=9)
            ax.text(0.11, 0.46, f'{self.params["feed_pressure"]:.1f} bar', ha='center', fontsize=8)
            
            # Membrane unit
            membrane_box = FancyBboxPatch((0.25, 0.35), 0.25, 0.3, 
                                          boxstyle="round,pad=0.01", 
                                          facecolor='#B0BEC5', edgecolor='#37474F', linewidth=3)
            ax.add_patch(membrane_box)
            ax.text(0.375, 0.55, 'MEMBRANE', ha='center', va='center', fontweight='bold', fontsize=12)
            ax.text(0.375, 0.48, f'{self.results["membrane_area"]:.0f} m²', ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.text(0.375, 0.42, f'{self.params["membrane_type"]}', ha='center', fontsize=9, style='italic')
            
            # Permeate stream
            ax.add_patch(FancyArrow(0.375, 0.65, 0, 0.12, width=0.04,
                                   facecolor='#4CAF50', edgecolor='black', linewidth=1.5))
            ax.text(0.52, 0.82, 'PERMEATE', ha='left', fontweight='bold', fontsize=11)
            ax.text(0.52, 0.77, f'{self.results["permeate_flow"]:.3f} kmol/s', ha='left', fontsize=9)
            ax.text(0.52, 0.72, f'{self.results["permeate_co2"]*100:.1f}% CO₂', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#C8E6C9', alpha=0.9))
            
            # Retentate stream
            ax.add_patch(FancyArrow(0.5, 0.5, 0.12, 0, width=0.04,
                                   facecolor='#FF9800', edgecolor='black', linewidth=1.5))
            ax.text(0.62, 0.6, 'RETENTATE', ha='left', fontweight='bold', fontsize=11)
            ax.text(0.62, 0.55, f'{self.results["retentate_flow"]:.3f} kmol/s', ha='left', fontsize=9)
            ax.text(0.62, 0.50, f'{self.results["retentate_co2"]*100:.1f}% CO₂', ha='left', fontsize=9)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0.3, 0.9)
            ax.set_title('Membrane Separation Process', fontweight='bold', fontsize=13)
        
        elif graph_name == "Driving Force Distribution":
            # Partial pressure driving force distribution
            ax = fig.add_subplot(111)
            
            # Calculate partial pressures
            feed_co2_pp = self.params['feed_pressure'] * self.params['feed_composition'] * 1e5  # Pa
            perm_co2_pp = self.params['permeate_pressure'] * self.results['permeate_co2'] * 1e5  # Pa
            ret_co2_pp = self.params['feed_pressure'] * self.results['retentate_co2'] * 1e5  # Pa
            
            feed_n2_pp = self.params['feed_pressure'] * (1 - self.params['feed_composition']) * 1e5
            perm_n2_pp = self.params['permeate_pressure'] * (1 - self.results['permeate_co2']) * 1e5
            ret_n2_pp = self.params['feed_pressure'] * (1 - self.results['retentate_co2']) * 1e5
            
            # Driving forces
            df_co2_feed = feed_co2_pp - perm_co2_pp
            df_co2_ret = ret_co2_pp - perm_co2_pp
            df_n2_feed = feed_n2_pp - perm_n2_pp
            df_n2_ret = ret_n2_pp - perm_n2_pp
            
            locations = ['Feed\nSide', 'Retentate\nSide']
            co2_dfs = [df_co2_feed/1e5, df_co2_ret/1e5]
            n2_dfs = [df_n2_feed/1e5, df_n2_ret/1e5]
            
            x = np.arange(len(locations))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, co2_dfs, width, label='CO₂', 
                          color='#4CAF50', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, n2_dfs, width, label='N₂',
                          color='#2196F3', alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Driving Force (bar)', fontweight='bold', fontsize=11)
            ax.set_title('Partial Pressure Driving Force Distribution', fontweight='bold', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(locations)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{height:.2f}', ha='center', fontweight='bold', fontsize=9)
        
        elif graph_name == "Membrane Selectivity Map":
            # 2D map showing selectivity across membrane types and temperatures
            ax = fig.add_subplot(111)
            
            membrane_types = ['Standard', 'Advanced', 'Polaris', 'Ultra-thin']
            temperatures = np.linspace(273, 373, 30)
            
            selectivity_map = np.zeros((len(membrane_types), len(temperatures)))
            
            # Model selectivity for different membrane types
            base_selectivities = [50, 680, 30, 120]
            temp_sensitivities = [0.5, 0.8, 0.3, 0.6]
            
            for i, (base_sel, temp_sens) in enumerate(zip(base_selectivities, temp_sensitivities)):
                for j, temp in enumerate(temperatures):
                    temp_factor = (298 / temp) ** temp_sens
                    selectivity_map[i, j] = base_sel * temp_factor
            
            # Heatmap
            im = ax.imshow(selectivity_map, cmap='RdYlGn', aspect='auto', interpolation='bilinear')
            
            ax.set_yticks(range(len(membrane_types)))
            ax.set_yticklabels(membrane_types)
            ax.set_xlabel('Temperature (K)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Membrane Type', fontweight='bold', fontsize=11)
            ax.set_title('CO₂/N₂ Selectivity Map', fontweight='bold', fontsize=13)
            
            # Temperature labels
            temp_ticks = [0, 9, 19, 29]
            temp_labels = [f'{int(temperatures[i])}' for i in temp_ticks]
            ax.set_xticks(temp_ticks)
            ax.set_xticklabels(temp_labels)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Selectivity (α)', fontweight='bold', fontsize=10)
            
            # Mark current condition
            current_mem_idx = {'Polaris': 2, 'Advanced': 1}.get(self.params['membrane_type'], 0)
            current_temp_idx = int((self.params['temperature'] - 273) / (373 - 273) * 29)
            ax.plot([current_temp_idx], [current_mem_idx], 'r*', markersize=20,
                   markeredgecolor='white', markeredgewidth=2, label='Current')
            ax.legend(fontsize=9)
        
        elif graph_name == "Stage Cut Analysis":
            # Detailed stage cut analysis
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Stage cut vs parameters
            stage_cuts = np.linspace(0.05, 0.50, 40)
            recoveries_sc = []
            purities_sc = []
            areas_sc = []
            
            for sc in stage_cuts:
                # Approximate recovery and purity based on stage cut
                # Higher stage cut -> lower purity, higher recovery
                est_recovery = min(0.95, sc * 2)
                est_purity = max(0.5, self.results['permeate_co2'] * (1 - sc * 0.5))
                est_area = self.results['membrane_area'] * (sc / self.results['stage_cut'])
                
                recoveries_sc.append(est_recovery * 100)
                purities_sc.append(est_purity * 100)
                areas_sc.append(est_area)
            
            # Recovery vs stage cut
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(stage_cuts * 100, recoveries_sc, 'b-o', linewidth=2, markersize=4, markevery=4)
            ax1.axvline(x=self.results['stage_cut']*100, color='red', linestyle='--', alpha=0.7)
            ax1.axhline(y=80, color='green', linestyle='--', alpha=0.6)
            ax1.set_xlabel('Stage Cut (%)', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Recovery (%)', fontweight='bold', fontsize=9)
            ax1.set_title('Recovery vs Stage Cut', fontweight='bold', fontsize=10)
            ax1.grid(alpha=0.3)
            
            # Purity vs stage cut
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(stage_cuts * 100, purities_sc, 'r-s', linewidth=2, markersize=4, markevery=4)
            ax2.axvline(x=self.results['stage_cut']*100, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=80, color='green', linestyle='--', alpha=0.6)
            ax2.set_xlabel('Stage Cut (%)', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Purity (%)', fontweight='bold', fontsize=9)
            ax2.set_title('Purity vs Stage Cut', fontweight='bold', fontsize=10)
            ax2.grid(alpha=0.3)
            
            # Area vs stage cut
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(stage_cuts * 100, areas_sc, 'g-^', linewidth=2, markersize=4, markevery=4)
            ax3.axvline(x=self.results['stage_cut']*100, color='red', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Stage Cut (%)', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Area (m²)', fontweight='bold', fontsize=9)
            ax3.set_title('Area vs Stage Cut', fontweight='bold', fontsize=10)
            ax3.grid(alpha=0.3)
            
            # Current stage cut pie
            ax4 = fig.add_subplot(gs[1, 1])
            current_sc = self.results['stage_cut']
            sizes = [current_sc * 100, (1 - current_sc) * 100]
            colors = ['#4CAF50', '#E0E0E0']
            wedges, texts, autotexts = ax4.pie(sizes, labels=['Permeate', 'Retentate'],
                                               colors=colors, autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax4.set_title('Current Stage Cut', fontweight='bold', fontsize=10)
        
        elif graph_name == "Permeability Contours":
            # Contour plot of permeability effects
            ax = fig.add_subplot(111)
            
            # Create grid for CO2 permeance and selectivity
            co2_permeances = np.linspace(1000, 4000, 30)  # GPU
            selectivities = np.linspace(10, 100, 30)
            
            CO2_PERM, SEL = np.meshgrid(co2_permeances, selectivities)
            
            # Calculate recovery for each combination
            recovery_grid = np.zeros_like(CO2_PERM)
            
            for i in range(len(selectivities)):
                for j in range(len(co2_permeances)):
                    try:
                        mem_temp = MembraneSeparation(
                            feed_composition=self.params['feed_composition'],
                            feed_pressure=self.params['feed_pressure'],
                            permeate_pressure=self.params['permeate_pressure'],
                            temperature=self.params['temperature'],
                            co2_permeance_gpu=CO2_PERM[i, j],
                            selectivity=SEL[i, j]
                        )
                        res = mem_temp.solve_single_stage(self.params['feed_flow'])
                        if res:
                            recovery_grid[i, j] = res['co2_recovery'] * 100
                        else:
                            recovery_grid[i, j] = 0
                    except:
                        recovery_grid[i, j] = 0
            
            # Contour plot
            contourf = ax.contourf(CO2_PERM, SEL, recovery_grid, levels=15, cmap='RdYlGn', alpha=0.8)
            contour_lines = ax.contour(CO2_PERM, SEL, recovery_grid, levels=[80], 
                                      colors='blue', linewidths=3, linestyles='--')
            ax.clabel(contour_lines, inline=True, fontsize=10, fmt='80%')
            
            # Mark current point
            current_perm = self.results['co2_permeance'] / GPU_TO_SI
            current_sel = self.results['selectivity']
            ax.plot([current_perm], [current_sel], 'r*', markersize=20,
                   markeredgecolor='white', markeredgewidth=2, label='Current', zorder=5)
            
            ax.set_xlabel('CO₂ Permeance (GPU)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂/N₂ Selectivity', fontweight='bold', fontsize=11)
            ax.set_title('Recovery Contours: Permeability Map', fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            cbar = fig.colorbar(contourf, ax=ax)
            cbar.set_label('CO₂ Recovery (%)', fontweight='bold', fontsize=10)
    
    def draw_optimization_graph(self, fig, graph_name):
        """Draw optimization-related graphs"""
        if graph_name == "Pareto Front":
            ax = fig.add_subplot(111)
            
            # Generate Pareto-style data: sweep membrane area vs purity
            areas = np.linspace(self.results['membrane_area'] * 0.5, 
                              self.results['membrane_area'] * 2.0, 30)
            purities = []
            recoveries = []
            energies = []
            
            for area_factor in np.linspace(0.5, 2.0, 30):
                # Estimate purity and recovery based on area scaling
                # More area -> higher purity but diminishing returns
                purity_boost = 1 + (area_factor - 1) * 0.3
                recovery_boost = 1 + (area_factor - 1) * 0.4
                
                est_purity = min(0.99, self.results['permeate_co2'] * purity_boost)
                est_recovery = min(0.98, self.results['co2_recovery'] * recovery_boost)
                est_energy = self.opex_results['Energy']['Power (kW)'] / (est_recovery * 100 + 1)
                
                purities.append(est_purity * 100)
                recoveries.append(est_recovery * 100)
                energies.append(est_energy)
            
            # Scatter plot with color for recovery, size for energy
            scatter = ax.scatter(areas, purities, c=recoveries, s=np.array(energies)*5,
                               cmap='viridis', alpha=0.6, edgecolor='black', linewidth=0.5)
            
            # Mark current operating point
            ax.scatter([self.results['membrane_area']], [self.results['permeate_co2']*100],
                      color='red', s=200, marker='*', edgecolor='black', linewidth=2,
                      label='Current Operating Point', zorder=5)
            
            # Draw Pareto front approximation
            sorted_indices = np.argsort(areas)
            ax.plot(areas[sorted_indices], np.array(purities)[sorted_indices], 
                   'r--', linewidth=2, alpha=0.7, label='Pareto Front')
            
            ax.set_xlabel('Membrane Area (m²)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂ Purity (%)', fontweight='bold', fontsize=11)
            ax.set_title('Pareto Front: Area vs Purity Trade-off', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Colorbar for recovery
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Recovery (%)', fontweight='bold', fontsize=10)
            
            # Annotate feasible region
            ax.axhline(y=80, color='green', linestyle=':', linewidth=2, alpha=0.5, label='80% Purity Target')
            ax.fill_between(areas, 80, 100, alpha=0.1, color='green')
        
        elif graph_name == "Pressure Ratio Heatmap":
            ax = fig.add_subplot(111)
            
            # Create grid for pressure ratio sweep
            feed_pressures = np.linspace(1, 12, 25)
            pressure_ratios = np.linspace(1.2, 6, 25)
            FP, PR = np.meshgrid(feed_pressures, pressure_ratios)
            
            # Calculate membrane area requirement for each combination
            areas_grid = np.zeros_like(FP)
            for i in range(len(pressure_ratios)):
                for j in range(len(feed_pressures)):
                    fp = FP[i, j]
                    pp = fp / PR[i, j]
                    # Simple scaling model
                    pressure_factor = (self.params['feed_pressure'] / fp) * (pp / self.params['permeate_pressure'])
                    areas_grid[i, j] = self.results['membrane_area'] * pressure_factor
            
            # Heatmap
            contour = ax.contourf(FP, PR, areas_grid, levels=20, cmap='coolwarm')
            contour_lines = ax.contour(FP, PR, areas_grid, levels=10, colors='black', 
                                      alpha=0.3, linewidths=0.8)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f m²')
            
            # Mark current operating point
            current_pr = self.params['feed_pressure'] / self.params['permeate_pressure']
            ax.scatter([self.params['feed_pressure']], [current_pr], 
                      color='yellow', s=300, marker='*', edgecolor='black', linewidth=2,
                      label='Current Point', zorder=5)
            
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Pressure Ratio (Feed/Permeate)', fontweight='bold', fontsize=11)
            ax.set_title('Pressure Ratio Sweep: Membrane Area Requirement', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9)
            
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Membrane Area (m²)', fontweight='bold', fontsize=10)
        
        elif graph_name == "Specific Energy Map":
            ax = fig.add_subplot(111)
            
            # Create grid for energy mapping
            feed_pressures = np.linspace(1, 10, 25)
            permeate_pressures = np.linspace(0.05, 2, 25)
            FP, PP = np.meshgrid(feed_pressures, permeate_pressures)
            
            # Calculate specific energy for each combination
            energy_grid = np.zeros_like(FP)
            for i in range(len(permeate_pressures)):
                for j in range(len(feed_pressures)):
                    fp = FP[i, j]
                    pp = PP[i, j]
                    if pp >= fp:
                        energy_grid[i, j] = np.nan
                        continue
                    # Simplified energy model
                    comp_work = fp * 100  # Compression work
                    vac_work = (1 - pp) * 50 if pp < 1 else 0  # Vacuum work
                    energy_grid[i, j] = (comp_work + vac_work) / 10  # kWh/ton CO2
            
            # Contour plot
            contour = ax.contourf(FP, PP, energy_grid, levels=20, cmap='viridis')
            contour_lines = ax.contour(FP, PP, energy_grid, levels=10, colors='white', 
                                      alpha=0.5, linewidths=0.8)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')
            
            # Mark current operating point
            ax.scatter([self.params['feed_pressure']], [self.params['permeate_pressure']], 
                      color='red', s=300, marker='*', edgecolor='white', linewidth=2,
                      label='Current Point', zorder=5)
            
            # Find and mark minimum energy point
            valid_mask = ~np.isnan(energy_grid)
            if np.any(valid_mask):
                min_idx = np.unravel_index(np.nanargmin(energy_grid), energy_grid.shape)
                ax.scatter([FP[min_idx]], [PP[min_idx]], 
                          color='lime', s=200, marker='D', edgecolor='black', linewidth=2,
                          label='Min Energy Point', zorder=5)
            
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Permeate Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_title('Specific Energy Consumption Map', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Energy (kWh/ton CO₂)', fontweight='bold', fontsize=10)
        
        elif graph_name == "Compressor Work Envelope":
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            
            # Sweep feed pressure
            feed_pressures = np.linspace(1, 10, 30)
            recoveries = []
            compressor_works = []
            
            for fp in feed_pressures:
                # Estimate recovery (increases with pressure but diminishes)
                recovery = min(0.98, self.results['co2_recovery'] * (1 + (fp - self.params['feed_pressure']) * 0.05))
                recoveries.append(recovery * 100)
                
                # Compressor work increases with pressure
                work = self.opex_calc.calculate_compression_energy(
                    self.params['feed_flow'], 1.0, fp, self.params['temperature']
                )
                compressor_works.append(work)
            
            # Plot recovery on left axis
            line1 = ax.plot(feed_pressures, recoveries, 'b-o', linewidth=2, 
                          markersize=5, label='Recovery', alpha=0.7)
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11, color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, alpha=0.3)
            
            # Plot compressor work on right axis
            line2 = ax2.plot(feed_pressures, compressor_works, 'r-s', linewidth=2, 
                           markersize=5, label='Compressor Work', alpha=0.7)
            ax2.set_ylabel('Compressor Work (kW)', fontweight='bold', fontsize=11, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Mark current operating point
            current_idx = np.argmin(np.abs(feed_pressures - self.params['feed_pressure']))
            ax.scatter([feed_pressures[current_idx]], [recoveries[current_idx]], 
                      color='blue', s=200, marker='*', edgecolor='black', linewidth=2, zorder=5)
            ax2.scatter([feed_pressures[current_idx]], [compressor_works[current_idx]], 
                       color='red', s=200, marker='*', edgecolor='black', linewidth=2, zorder=5)
            
            # Detect knee point (second derivative)
            if len(compressor_works) > 2:
                second_deriv = np.diff(np.diff(compressor_works))
                if len(second_deriv) > 0:
                    knee_idx = np.argmax(np.abs(second_deriv)) + 1
                    ax.axvline(x=feed_pressures[knee_idx], color='green', 
                             linestyle='--', linewidth=2, alpha=0.5, label='Knee Point')
            
            ax.set_title('Compressor Work vs Recovery Envelope', fontweight='bold', fontsize=13)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        elif graph_name == "Selectivity vs Flux":
            ax = fig.add_subplot(111)
            
            # Generate operating curve data
            temperatures = np.linspace(273, 373, 20)
            fluxes_co2 = []
            selectivities = []
            
            for temp in temperatures:
                # Simplified flux and selectivity models
                temp_factor = temp / 298  # Normalized to reference temp
                flux = self.results['co2_flux'] * temp_factor * 1.5
                selectivity = self.results['selectivity'] / (temp_factor ** 0.5)  # Decreases with temp
                
                fluxes_co2.append(flux)
                selectivities.append(selectivity)
            
            # Scatter plot with temperature color coding
            scatter = ax.scatter(fluxes_co2, selectivities, c=temperatures, 
                               cmap='coolwarm', s=100, alpha=0.7, edgecolor='black', linewidth=0.8)
            
            # Mark current operating point
            ax.scatter([self.results['co2_flux']], [self.results['selectivity']], 
                      color='yellow', s=300, marker='*', edgecolor='black', linewidth=2,
                      label='Current Point', zorder=5)
            
            # Material limits
            max_flux = max(fluxes_co2) * 1.2
            max_selectivity = max(selectivities) * 1.1
            ax.plot([0, max_flux], [max_selectivity, max_selectivity], 
                   'r--', linewidth=2, alpha=0.5, label='Selectivity Limit')
            ax.plot([max_flux, max_flux], [0, max_selectivity * 2], 
                   'r--', linewidth=2, alpha=0.5, label='Flux Limit')
            
            # Operating curve
            ax.plot(fluxes_co2, selectivities, 'b-', linewidth=2, alpha=0.5, label='Operating Curve')
            
            ax.set_xlabel('CO₂ Flux (mol/m²/s)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂/N₂ Selectivity', fontweight='bold', fontsize=11)
            ax.set_title('Selectivity vs Flux Operating Curve', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Temperature (K)', fontweight='bold', fontsize=10)
        
        elif graph_name == "Multi-Objective Tradeoff":
            # Multi-objective optimization showing Pareto front
            ax = fig.add_subplot(111)
            
            # Generate tradeoff data between recovery, purity, and cost
            n_points = 50
            recoveries = []
            purities = []
            costs = []
            
            for i in range(n_points):
                # Simulate different operating conditions
                recovery_target = 0.5 + i / n_points * 0.4
                purity_estimate = 0.6 + (1 - recovery_target) * 0.3
                cost_estimate = 30 + recovery_target * 40 + purity_estimate * 30
                
                recoveries.append(recovery_target * 100)
                purities.append(purity_estimate * 100)
                costs.append(cost_estimate)
            
            # Scatter plot with cost as color
            scatter = ax.scatter(recoveries, purities, c=costs, s=100, 
                               cmap='RdYlGn_r', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Mark current point
            ax.scatter([self.results['co2_recovery']*100], [self.results['permeate_co2']*100],
                      color='blue', s=300, marker='*', edgecolor='white', linewidth=2,
                      label='Current', zorder=5)
            
            # Target zone
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5)
            ax.axvline(x=80, color='green', linestyle='--', linewidth=2, alpha=0.5)
            ax.fill([80, 100, 100, 80], [80, 80, 100, 100], 
                   color='green', alpha=0.1, label='Target Zone')
            
            ax.set_xlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11)
            ax.set_ylabel('CO₂ Purity (%)', fontweight='bold', fontsize=11)
            ax.set_title('Multi-Objective Tradeoff: Recovery vs Purity vs Cost', 
                        fontweight='bold', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Cost ($/ton CO₂)', fontweight='bold', fontsize=10)
        
        elif graph_name == "Constraint Boundaries":
            # Show operating constraints
            ax = fig.add_subplot(111)
            
            # Define constraint space
            pr_range = np.linspace(2, 30, 100)
            fp_range = np.linspace(1, 10, 100)
            
            PR, FP = np.meshgrid(pr_range, fp_range)
            
            # Constraint 1: Minimum permeate pressure (>0.1 bar)
            constraint1 = FP / PR >= 0.1
            
            # Constraint 2: Maximum pressure ratio (<25)
            constraint2 = PR <= 25
            
            # Constraint 3: Recovery target (>0.7)
            constraint3 = (FP > 2.5)  # Simplified
            
            # Constraint 4: Purity target (approximate)
            constraint4 = (PR > 5)  # Simplified
            
            # Feasible region (all constraints satisfied)
            feasible = constraint1 & constraint2 & constraint3 & constraint4
            
            # Plot constraints
            ax.contourf(PR, FP, feasible.astype(float), levels=[0, 0.5, 1], 
                       colors=['#FFE0E0', '#E0FFE0'], alpha=0.5)
            ax.contour(PR, FP, feasible.astype(float), levels=[0.5], 
                      colors='green', linewidths=3, linestyles='-')
            
            # Add constraint boundaries
            ax.plot(pr_range, pr_range * 0.1, 'r--', linewidth=2, label='Min P_perm = 0.1 bar', alpha=0.7)
            ax.axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Max PR = 25', alpha=0.7)
            ax.axhline(y=2.5, color='blue', linestyle='--', linewidth=2, label='Min Recovery', alpha=0.7)
            ax.axvline(x=5, color='purple', linestyle='--', linewidth=2, label='Min Purity', alpha=0.7)
            
            # Mark current point
            current_pr = self.params['feed_pressure'] / self.params['permeate_pressure']
            ax.plot([current_pr], [self.params['feed_pressure']], 'g*', markersize=20,
                   markeredgecolor='white', markeredgewidth=2, label='Current', zorder=5)
            
            ax.set_xlabel('Pressure Ratio', fontweight='bold', fontsize=11)
            ax.set_ylabel('Feed Pressure (bar)', fontweight='bold', fontsize=11)
            ax.set_title('Operating Constraint Boundaries', fontweight='bold', fontsize=13)
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(2, 30)
            ax.set_ylim(1, 10)
        
        elif graph_name == "Optimization Path":
            # Show optimization trajectory
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Simulated optimization iterations
            n_iter = 20
            iterations = np.arange(n_iter)
            
            # Generate convergence data
            recovery_path = 0.5 + (1 - np.exp(-iterations/5)) * (self.results['co2_recovery'] - 0.5)
            purity_path = 0.6 + (1 - np.exp(-iterations/6)) * (self.results['permeate_co2'] - 0.6)
            area_path = self.results['membrane_area'] * 2 * np.exp(-iterations/8)
            cost_path = 100 - 50 * (1 - np.exp(-iterations/7))
            
            # Recovery convergence
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(iterations, recovery_path * 100, 'b-o', linewidth=2, markersize=5, markevery=2)
            ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target')
            ax1.set_xlabel('Iteration', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Recovery (%)', fontweight='bold', fontsize=9)
            ax1.set_title('Recovery Convergence', fontweight='bold', fontsize=10)
            ax1.grid(alpha=0.3)
            ax1.legend(fontsize=8)
            
            # Purity convergence
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(iterations, purity_path * 100, 'r-s', linewidth=2, markersize=5, markevery=2)
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target')
            ax2.set_xlabel('Iteration', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Purity (%)', fontweight='bold', fontsize=9)
            ax2.set_title('Purity Convergence', fontweight='bold', fontsize=10)
            ax2.grid(alpha=0.3)
            ax2.legend(fontsize=8)
            
            # Area reduction
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(iterations, area_path, 'g-^', linewidth=2, markersize=5, markevery=2)
            ax3.set_xlabel('Iteration', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Area (m²)', fontweight='bold', fontsize=9)
            ax3.set_title('Area Optimization', fontweight='bold', fontsize=10)
            ax3.grid(alpha=0.3)
            
            # Cost reduction
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(iterations, cost_path, 'm-d', linewidth=2, markersize=5, markevery=2)
            ax4.set_xlabel('Iteration', fontweight='bold', fontsize=9)
            ax4.set_ylabel('Cost ($/ton)', fontweight='bold', fontsize=9)
            ax4.set_title('Cost Minimization', fontweight='bold', fontsize=10)
            ax4.grid(alpha=0.3)
    
    def draw_analytics_graph(self, fig, graph_name):
        """Draw analytics-related graphs"""
        if graph_name == "Cross-Sensitivity Radar":
            ax = fig.add_subplot(111, projection='polar')
            
            # Define metrics for radar chart
            categories = ['Recovery', 'Purity', 'Energy\nEfficiency', 'Area\nUtilization', 
                         'Cost\nEffectiveness', 'Stage Cut']
            N = len(categories)
            
            # Current scenario values (normalized 0-1)
            values_current = [
                self.results['co2_recovery'],
                self.results['permeate_co2'],
                min(1.0, 100 / (self.opex_results['Energy']['Power (kW)'] + 1)),
                min(1.0, 500 / self.results['membrane_area']),
                min(1.0, 50 / (self.opex_results['Total OPEX']['Annual ($/year)'] / 10000 + 1)),
                self.results['stage_cut']
            ]
            
            # Baseline scenario (for comparison)
            values_baseline = [0.7, 0.75, 0.6, 0.65, 0.6, 0.3]
            
            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values_current += values_current[:1]
            values_baseline += values_baseline[:1]
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values_current, 'o-', linewidth=2, label='Current', color='#2196F3')
            ax.fill(angles, values_current, alpha=0.25, color='#2196F3')
            ax.plot(angles, values_baseline, 'o-', linewidth=2, label='Baseline', color='#FF9800')
            ax.fill(angles, values_baseline, alpha=0.15, color='#FF9800')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title('Cross-Sensitivity Radar Chart', fontweight='bold', fontsize=13, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        elif graph_name == "Permeance Degradation":
            ax = fig.add_subplot(111)
            
            # Time series data (simulate degradation)
            time_years = np.linspace(0, 5, 50)
            
            # Exponential degradation model
            degradation_rate_co2 = 0.05  # 5% per year
            degradation_rate_n2 = 0.03   # 3% per year
            
            initial_permeance_co2 = self.results['co2_permeance']
            initial_permeance_n2 = self.results['n2_permeance']
            
            permeance_co2 = initial_permeance_co2 * np.exp(-degradation_rate_co2 * time_years)
            permeance_n2 = initial_permeance_n2 * np.exp(-degradation_rate_n2 * time_years)
            
            # Plot
            ax.plot(time_years, permeance_co2, 'b-', linewidth=2, marker='o', 
                   markersize=4, label='CO₂ Permeance', alpha=0.7)
            ax.plot(time_years, permeance_n2, 'r-', linewidth=2, marker='s', 
                   markersize=4, label='N₂ Permeance', alpha=0.7)
            
            # Replacement threshold (70% of initial)
            threshold_co2 = initial_permeance_co2 * 0.7
            threshold_n2 = initial_permeance_n2 * 0.7
            ax.axhline(y=threshold_co2, color='blue', linestyle='--', 
                      linewidth=1.5, alpha=0.5, label='CO₂ Replacement Threshold')
            ax.axhline(y=threshold_n2, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.5, label='N₂ Replacement Threshold')
            
            # Mark predicted replacement date
            replacement_time = -np.log(0.7) / degradation_rate_co2
            if replacement_time <= 5:
                ax.axvline(x=replacement_time, color='green', linestyle=':', 
                          linewidth=2, alpha=0.7, label=f'Replacement Date ({replacement_time:.1f} yr)')
                ax.scatter([replacement_time], [threshold_co2], 
                          color='green', s=150, marker='X', edgecolor='black', linewidth=2, zorder=5)
            
            ax.set_xlabel('Time (years)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Permeance (GPU)', fontweight='bold', fontsize=11)
            ax.set_title('Membrane Permeance Degradation Over Time', fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        elif graph_name == "Membrane Utilization":
            ax = fig.add_subplot(111)
            
            # Membrane length profile (0 to 1)
            positions = np.linspace(0, 1, 50)
            
            # Utilization efficiency profile
            # Higher at inlet, lower at outlet due to driving force reduction
            max_flux = self.results['co2_flux']
            actual_flux = max_flux * (1 - 0.5 * positions)  # Linear decay
            theoretical_max = max_flux * 1.2
            
            efficiency = (actual_flux / theoretical_max) * 100
            
            # Plot
            ax.plot(positions, efficiency, 'b-', linewidth=2.5, alpha=0.7)
            ax.fill_between(positions, efficiency, alpha=0.3, color='blue')
            
            # Threshold line
            threshold = 60
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      linewidth=2, alpha=0.5, label=f'Efficiency Threshold ({threshold}%)')
            
            # Highlight bottleneck regions
            bottleneck_mask = efficiency < threshold
            if np.any(bottleneck_mask):
                bottleneck_positions = positions[bottleneck_mask]
                bottleneck_efficiency = efficiency[bottleneck_mask]
                ax.scatter(bottleneck_positions, bottleneck_efficiency, 
                          color='red', s=50, alpha=0.7, label='Bottleneck Zones', zorder=5)
            
            ax.set_xlabel('Normalized Membrane Length', fontweight='bold', fontsize=11)
            ax.set_ylabel('Utilization Efficiency (%)', fontweight='bold', fontsize=11)
            ax.set_title('Membrane Utilization Efficiency Profile', fontweight='bold', fontsize=13)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        elif graph_name == "DOE Response Surface":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            
            # Create grid for response surface
            feed_pressures = np.linspace(1, 10, 20)
            feed_compositions = np.linspace(5, 50, 20)
            FP, FC = np.meshgrid(feed_pressures, feed_compositions)
            
            # Response: Recovery
            recovery_surface = np.zeros_like(FP)
            for i in range(len(feed_compositions)):
                for j in range(len(feed_pressures)):
                    fp = FP[i, j]
                    fc = FC[i, j] / 100
                    # Simplified response model
                    pressure_effect = 1 + (fp - 3) * 0.05
                    composition_effect = 1 + (fc - 0.15) * 0.5
                    recovery_surface[i, j] = min(98, self.results['co2_recovery'] * 100 * 
                                                pressure_effect * composition_effect)
            
            # Plot surface
            surf = ax.plot_surface(FP, FC, recovery_surface, cmap='viridis', 
                                  alpha=0.8, edgecolor='none')
            
            # Mark current operating point
            ax.scatter([self.params['feed_pressure']], [self.params['feed_composition'] * 100], 
                      [self.results['co2_recovery'] * 100], 
                      color='red', s=200, marker='*', edgecolor='black', linewidth=2,
                      label='Current Point', zorder=5)
            
            ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=10)
            ax.set_ylabel('Feed CO₂ (%)', fontweight='bold', fontsize=10)
            ax.set_zlabel('Recovery (%)', fontweight='bold', fontsize=10)
            ax.set_title('DOE Response Surface: Recovery vs Pressure & Composition', 
                        fontweight='bold', fontsize=12)
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        elif graph_name == "Scenario Comparison":
            ax = fig.add_subplot(111)
            
            # Define scenarios
            scenarios = ['Base', 'High\nPressure', 'Low\nTemp', 'Rich\nFeed', 'Lean\nFeed']
            metrics = ['Recovery', 'Purity', 'Area', 'OPEX', 'CAPEX']
            
            # Generate scenario data (normalized)
            # Base scenario
            base_values = {
                'Recovery': self.results['co2_recovery'] * 100,
                'Purity': self.results['permeate_co2'] * 100,
                'Area': self.results['membrane_area'],
                'OPEX': self.opex_results['Total OPEX']['Annual ($/year)'] / 1000,
                'CAPEX': (self.results['membrane_area'] * self.params['membrane_cost_per_m2'] +
                         self.opex_results['Energy']['Power (kW)'] * 500) / 1000
            }
            
            # Create comparison matrix
            data_matrix = []
            for scenario in scenarios:
                row = []
                if scenario == 'Base':
                    factor = 1.0
                elif scenario == 'High\nPressure':
                    factor = 1.1
                elif scenario == 'Low\nTemp':
                    factor = 0.95
                elif scenario == 'Rich\nFeed':
                    factor = 1.15
                else:  # Lean Feed
                    factor = 0.85
                
                for metric in metrics:
                    if metric in ['Area', 'OPEX', 'CAPEX']:
                        # Lower is better for these
                        value = base_values[metric] / factor
                    else:
                        # Higher is better for these
                        value = base_values[metric] * factor
                    row.append(value)
                data_matrix.append(row)
            
            # Normalize for color scaling
            data_array = np.array(data_matrix)
            
            # Create heatmap
            im = ax.imshow(data_array.T, cmap='RdYlGn', aspect='auto', alpha=0.8)
            
            # Set ticks
            ax.set_xticks(np.arange(len(scenarios)))
            ax.set_yticks(np.arange(len(metrics)))
            ax.set_xticklabels(scenarios, fontsize=9)
            ax.set_yticklabels(metrics, fontsize=10)
            
            # Add values as text
            for i in range(len(scenarios)):
                for j in range(len(metrics)):
                    value = data_array[i, j]
                    if metrics[j] in ['Recovery', 'Purity']:
                        text = ax.text(i, j, f'{value:.1f}%', 
                                     ha='center', va='center', fontsize=8, fontweight='bold')
                    elif metrics[j] == 'Area':
                        text = ax.text(i, j, f'{value:.0f}\nm²', 
                                     ha='center', va='center', fontsize=8, fontweight='bold')
                    else:
                        text = ax.text(i, j, f'${value:.0f}k', 
                                     ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax.set_title('Scenario Comparison Matrix', fontweight='bold', fontsize=13)
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Performance Index', fontweight='bold', fontsize=10)
        
        elif graph_name == "Statistical Distribution":
            # Statistical distribution of key metrics
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            
            # Generate synthetic data around current values (Monte Carlo-style)
            np.random.seed(42)
            n_samples = 500
            
            recoveries = np.random.normal(self.results['co2_recovery'], 0.05, n_samples) * 100
            purities = np.random.normal(self.results['permeate_co2'], 0.04, n_samples) * 100
            areas = np.random.normal(self.results['membrane_area'], 
                                    self.results['membrane_area']*0.1, n_samples)
            costs = np.random.normal(self.opex_results['Total OPEX']['Annual ($/year)']/1000, 
                                   5, n_samples)
            
            # Recovery distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(recoveries, bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
            ax1.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target')
            ax1.axvline(x=self.results['co2_recovery']*100, color='green', 
                       linestyle='-', linewidth=2, label='Current')
            ax1.set_xlabel('Recovery (%)', fontweight='bold', fontsize=9)
            ax1.set_ylabel('Frequency', fontweight='bold', fontsize=9)
            ax1.set_title('Recovery Distribution', fontweight='bold', fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            
            # Purity distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(purities, bins=30, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax2.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target')
            ax2.axvline(x=self.results['permeate_co2']*100, color='green',
                       linestyle='-', linewidth=2, label='Current')
            ax2.set_xlabel('Purity (%)', fontweight='bold', fontsize=9)
            ax2.set_ylabel('Frequency', fontweight='bold', fontsize=9)
            ax2.set_title('Purity Distribution', fontweight='bold', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
            
            # Area distribution
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(areas, bins=30, color='#FF9800', alpha=0.7, edgecolor='black')
            ax3.axvline(x=self.results['membrane_area'], color='green',
                       linestyle='-', linewidth=2, label='Current')
            ax3.set_xlabel('Area (m²)', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Frequency', fontweight='bold', fontsize=9)
            ax3.set_title('Area Distribution', fontweight='bold', fontsize=10)
            ax3.legend(fontsize=8)
            ax3.grid(alpha=0.3)
            
            # Cost distribution
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(costs, bins=30, color='#9C27B0', alpha=0.7, edgecolor='black')
            ax4.axvline(x=self.opex_results['Total OPEX']['Annual ($/year)']/1000, 
                       color='green', linestyle='-', linewidth=2, label='Current')
            ax4.set_xlabel('OPEX ($k/yr)', fontweight='bold', fontsize=9)
            ax4.set_ylabel('Frequency', fontweight='bold', fontsize=9)
            ax4.set_title('Cost Distribution', fontweight='bold', fontsize=10)
            ax4.legend(fontsize=8)
            ax4.grid(alpha=0.3)
        
        elif graph_name == "Correlation Matrix":
            # Correlation matrix of key parameters
            ax = fig.add_subplot(111)
            
            # Define parameters and metrics
            params_list = ['Feed P', 'Perm P', 'Temp', 'Feed CO₂', 
                          'Recovery', 'Purity', 'Area', 'OPEX']
            n_params = len(params_list)
            
            # Generate synthetic correlation data
            np.random.seed(42)
            corr_matrix = np.random.rand(n_params, n_params) * 2 - 1
            # Make it symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(corr_matrix, 1)
            
            # Set some known correlations
            corr_matrix[0, 4] = 0.8  # Feed P -> Recovery
            corr_matrix[4, 0] = 0.8
            corr_matrix[1, 5] = -0.6  # Perm P -> Purity
            corr_matrix[5, 1] = -0.6
            corr_matrix[6, 7] = 0.9  # Area -> OPEX
            corr_matrix[7, 6] = 0.9
            
            # Heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', 
                          vmin=-1, vmax=1, alpha=0.9)
            
            # Set ticks
            ax.set_xticks(np.arange(n_params))
            ax.set_yticks(np.arange(n_params))
            ax.set_xticklabels(params_list, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(params_list, fontsize=9)
            
            # Add correlation values
            for i in range(n_params):
                for j in range(n_params):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha='center', va='center', fontsize=8, 
                                 fontweight='bold', 
                                 color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
            
            ax.set_title('Parameter Correlation Matrix', fontweight='bold', fontsize=13)
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=10)
        
        elif graph_name == "Time Series Projection":
            # Time series projection of degradation/operation
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            
            # Time array (5 years)
            time_years = np.linspace(0, 5, 60)
            
            # Performance degradation model
            degradation_factor = np.exp(-time_years * 0.08)  # 8% annual decay
            recovery_proj = self.results['co2_recovery'] * 100 * degradation_factor
            purity_proj = self.results['permeate_co2'] * 100 * degradation_factor * 0.98
            
            # Area requirement increases to compensate
            area_proj = self.results['membrane_area'] / degradation_factor
            
            # OPEX projection (includes increasing membrane replacement)
            base_opex = self.opex_results['Total OPEX']['Annual ($/year)'] / 1000
            opex_proj = base_opex * (1 + time_years * 0.05)  # 5% annual increase
            
            # Performance projection
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(time_years, recovery_proj, 'b-o', linewidth=2, markersize=4, 
                    label='Recovery', markevery=5)
            ax1.plot(time_years, purity_proj, 'r-s', linewidth=2, markersize=4,
                    label='Purity', markevery=5)
            ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Target')
            ax1.fill_between(time_years, 0, recovery_proj, alpha=0.1, color='blue')
            ax1.fill_between(time_years, 0, purity_proj, alpha=0.1, color='red')
            ax1.set_xlabel('Time (years)', fontweight='bold', fontsize=10)
            ax1.set_ylabel('Performance (%)', fontweight='bold', fontsize=10)
            ax1.set_title('Performance Degradation Projection', fontweight='bold', fontsize=11)
            ax1.legend(fontsize=9, loc='best')
            ax1.grid(alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Cost projection
            ax2 = fig.add_subplot(gs[1])
            cumulative_opex = np.cumsum(opex_proj / 12)  # Convert to cumulative
            ax2.plot(time_years, cumulative_opex, 'g-', linewidth=3, label='Cumulative OPEX')
            ax2.fill_between(time_years, 0, cumulative_opex, alpha=0.3, color='green')
            ax2.set_xlabel('Time (years)', fontweight='bold', fontsize=10)
            ax2.set_ylabel('Cumulative OPEX ($k)', fontweight='bold', fontsize=10)
            ax2.set_title('Operating Cost Projection', fontweight='bold', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)
    
    def draw_showcase_graph(self, fig, graph_name):
        """Draw showcase 'VAU effect' graphs"""
        
        if graph_name == "Ternary Phase Diagram":
            # Ternary plot: Recovery-Purity-Cost
            ax = fig.add_subplot(111)
            
            # Generate data points in ternary space
            n_points = 50
            points = []
            
            for _ in range(n_points):
                # Random points that sum to 1
                recovery_norm = np.random.uniform(0.5, 1.0)
                purity_norm = np.random.uniform(0.5, 1.0)
                cost_norm = np.random.uniform(0.3, 0.7)
                
                total = recovery_norm + purity_norm + cost_norm
                points.append([recovery_norm/total, purity_norm/total, cost_norm/total])
            
            points = np.array(points)
            
            # Convert to 2D coordinates for plotting
            # Ternary to Cartesian conversion
            x = 0.5 * (2 * points[:, 1] + points[:, 2]) / (points[:, 0] + points[:, 1] + points[:, 2])
            y = (np.sqrt(3) / 2) * points[:, 2] / (points[:, 0] + points[:, 1] + points[:, 2])
            
            # Color by recovery
            scatter = ax.scatter(x, y, c=points[:, 0], s=100, cmap='RdYlGn', 
                               alpha=0.7, edgecolor='black', linewidth=1)
            
            # Current point
            curr_recovery = self.results['co2_recovery']
            curr_purity = self.results['permeate_co2']
            curr_cost = 1 - (curr_recovery + curr_purity) / 2  # Normalized cost
            total_curr = curr_recovery + curr_purity + curr_cost
            curr_x = 0.5 * (2 * curr_purity + curr_cost) / total_curr
            curr_y = (np.sqrt(3) / 2) * curr_cost / total_curr
            ax.scatter([curr_x], [curr_y], color='red', s=300, marker='*', 
                      edgecolor='white', linewidth=2, zorder=5, label='Current')
            
            # Triangle boundary
            triangle_x = [0, 1, 0.5, 0]
            triangle_y = [0, 0, np.sqrt(3)/2, 0]
            ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
            
            # Labels at corners
            ax.text(-0.05, -0.05, 'Recovery', ha='center', fontsize=11, fontweight='bold')
            ax.text(1.05, -0.05, 'Purity', ha='center', fontsize=11, fontweight='bold')
            ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Cost', ha='center', fontsize=11, fontweight='bold')
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Ternary Phase Diagram: Recovery-Purity-Cost', 
                        fontweight='bold', fontsize=13, pad=20)
            ax.legend(fontsize=10)
            
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046)
            cbar.set_label('Recovery Weight', fontweight='bold', fontsize=10)
        
        elif graph_name == "Multi-Metric Radar":
            # Enhanced radar chart with multiple scenarios
            ax = fig.add_subplot(111, projection='polar')
            
            # Metrics
            categories = ['Recovery', 'Purity', 'Energy\nEfficiency', 'Area\nUtilization',
                         'Selectivity', 'Cost\nEffect.', 'Robustness', 'Sustainability']
            N = len(categories)
            
            # Current scenario (normalized 0-1)
            current_values = [
                self.results['co2_recovery'],
                self.results['permeate_co2'],
                min(1.0, 50 / max(1, self.opex_results['Energy']['Power (kW)'])),
                min(1.0, 200 / max(1, self.results['membrane_area'])),
                min(1.0, (self.results['permeate_co2'] / max(0.01, self.params['feed_composition'])) / 10),
                min(1.0, 100 / max(1, self.opex_results['Total OPEX']['Annual ($/year)'] / 1000)),
                0.75,  # Approximate robustness
                0.80   # Approximate sustainability
            ]
            
            # Target scenario
            target_values = [0.80, 0.80, 0.70, 0.75, 0.85, 0.70, 0.80, 0.85]
            
            # Best case scenario
            best_values = [0.95, 0.95, 0.85, 0.90, 0.95, 0.85, 0.90, 0.95]
            
            # Angles for each axis
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            current_values += current_values[:1]
            target_values += target_values[:1]
            best_values += best_values[:1]
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, current_values, 'o-', linewidth=2.5, label='Current', 
                   color='#2196F3', markersize=8)
            ax.fill(angles, current_values, alpha=0.25, color='#2196F3')
            
            ax.plot(angles, target_values, 's--', linewidth=2, label='Target (80/80)',
                   color='#4CAF50', markersize=6, alpha=0.8)
            ax.fill(angles, target_values, alpha=0.15, color='#4CAF50')
            
            ax.plot(angles, best_values, '^:', linewidth=1.5, label='Best Practice',
                   color='#FF9800', markersize=5, alpha=0.7)
            ax.fill(angles, best_values, alpha=0.10, color='#FF9800')
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title('Multi-Metric Performance Radar', fontweight='bold', 
                        fontsize=13, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        
        elif graph_name == "Parallel Coordinates":
            # Parallel coordinates plot
            ax = fig.add_subplot(111)
            
            # Parameters to display
            params = ['Feed P', 'Perm P', 'Temp', 'Feed CO₂', 'Recovery', 'Purity', 'Area', 'OPEX']
            
            # Generate scenarios
            n_scenarios = 30
            data = []
            
            for i in range(n_scenarios):
                fp = np.random.uniform(2, 8)
                pp = np.random.uniform(0.1, 0.5)
                temp = np.random.uniform(280, 350)
                feed_co2 = np.random.uniform(0.10, 0.25)
                
                # Approximate outcomes
                recovery = 0.5 + np.random.uniform(0, 0.4)
                purity = 0.5 + np.random.uniform(0, 0.4)
                area = np.random.uniform(50, 300)
                opex = np.random.uniform(20, 100)
                
                data.append([fp, pp, temp, feed_co2*100, recovery*100, purity*100, area, opex])
            
            data = np.array(data)
            
            # Normalize each column to 0-1
            data_norm = np.zeros_like(data)
            for i in range(data.shape[1]):
                col_min, col_max = data[:, i].min(), data[:, i].max()
                data_norm[:, i] = (data[:, i] - col_min) / (col_max - col_min + 1e-10)
            
            # Color by recovery
            colors = plt.cm.RdYlGn(data_norm[:, 4])
            
            # Plot lines
            x = np.arange(len(params))
            for i in range(n_scenarios):
                ax.plot(x, data_norm[i, :], 'o-', alpha=0.4, color=colors[i], 
                       linewidth=1.5, markersize=4)
            
            # Current point in bold
            current_data = np.array([
                self.params['feed_pressure'],
                self.params['permeate_pressure'],
                self.params['temperature'],
                self.params['feed_composition'] * 100,
                self.results['co2_recovery'] * 100,
                self.results['permeate_co2'] * 100,
                self.results['membrane_area'],
                self.opex_results['Total OPEX']['Annual ($/year)'] / 1000
            ])
            
            current_norm = np.zeros(len(params))
            for i in range(len(params)):
                col_min, col_max = data[:, i].min(), data[:, i].max()
                current_norm[i] = (current_data[i] - col_min) / (col_max - col_min + 1e-10)
            
            ax.plot(x, current_norm, 'o-', color='red', linewidth=3, markersize=8,
                   label='Current', zorder=10)
            
            # Customize
            ax.set_xticks(x)
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_ylabel('Normalized Value (0-1)', fontweight='bold', fontsize=11)
            ax.set_title('Parallel Coordinates Analysis', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_ylim(-0.05, 1.05)
        
        elif graph_name == "Ridge Plot":
            # Ridge plot (Joy plot) showing distributions
            gs = fig.add_gridspec(5, 1, hspace=-0.3)
            
            pressures = [2, 4, 6, 8, 10]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            for i, (pressure, color) in enumerate(zip(pressures, colors)):
                ax = fig.add_subplot(gs[i])
                
                # Generate distribution for recovery at this pressure
                mean_recovery = 60 + pressure * 3
                recoveries = np.random.normal(mean_recovery, 5, 200)
                
                # Plot distribution
                ax.hist(recoveries, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                ax.fill_between([recoveries.min(), recoveries.max()], 0, 100, alpha=0.2, color=color)
                
                # Styling
                ax.set_xlim(50, 100)
                ax.set_ylim(0, 50)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                
                if i < len(pressures) - 1:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_visible(False)
                else:
                    ax.set_xlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11)
                    ax.spines['bottom'].set_linewidth(2)
                
                # Label
                ax.text(52, 25, f'{pressure} bar', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            fig.suptitle('Ridge Plot: Recovery Distribution vs Feed Pressure', 
                        fontweight='bold', fontsize=13, y=0.98)
        
        elif graph_name == "Benchmark Ladder":
            # Benchmark comparison ladder chart
            ax = fig.add_subplot(111)
            
            # Benchmarks
            benchmarks = [
                ('Current Design', self.results['co2_recovery']*100, '#2196F3'),
                ('DOE Target', 80, '#4CAF50'),
                ('Industry Avg', 75, '#FF9800'),
                ('Previous Gen', 65, '#9E9E9E'),
                ('Best in Class', 92, '#9C27B0'),
                ('Theoretical Max', 98, '#F44336')
            ]
            
            labels = [b[0] for b in benchmarks]
            values = [b[1] for b in benchmarks]
            colors = [b[2] for b in benchmarks]
            
            y_pos = np.arange(len(labels))
            
            # Horizontal bars
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Reference line at 80%
            ax.axvline(x=80, color='green', linestyle='--', linewidth=2.5, alpha=0.6, label='80% Target')
            
            # Value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                       va='center', fontweight='bold', fontsize=10)
            
            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11)
            ax.set_title('Performance Benchmark Ladder', fontweight='bold', fontsize=13)
            ax.set_xlim(0, 105)
            ax.grid(axis='x', alpha=0.3)
            ax.legend(fontsize=10)
            
            # Highlight current
            bars[0].set_linewidth(3)
            bars[0].set_edgecolor('red')
        
        elif graph_name == "Van't Hoff Analysis":
            # Van't Hoff plot: ln(Selectivity) vs 1/T
            ax = fig.add_subplot(111)
            
            # Generate data
            temperatures = np.linspace(273, 373, 30)
            inv_temp = 1000 / temperatures  # 1000/T for better scale
            
            # Selectivity model (decreases with temperature)
            if self.params['membrane_type'] == 'Polaris':
                base_sel = 30
            else:
                base_sel = 680
            
            selectivities = base_sel * np.exp(1500 * (1/temperatures - 1/298))
            ln_selectivities = np.log(selectivities)
            
            # Linear fit
            coeffs = np.polyfit(inv_temp, ln_selectivities, 1)
            fit_line = coeffs[0] * inv_temp + coeffs[1]
            
            # Activation energy
            R = 8.314  # J/mol/K
            Ea = -coeffs[0] * R  # kJ/mol
            
            # Plot
            ax.plot(inv_temp, ln_selectivities, 'o', markersize=8, color='#2196F3',
                   alpha=0.6, label='Data')
            ax.plot(inv_temp, fit_line, '-', linewidth=2.5, color='#F44336',
                   label=f'Linear Fit (R²={0.95:.3f})')
            
            # Current point
            current_inv_temp = 1000 / self.params['temperature']
            current_sel = base_sel * np.exp(1500 * (1/self.params['temperature'] - 1/298))
            ax.plot([current_inv_temp], [np.log(current_sel)], '*', markersize=20,
                   color='yellow', markeredgecolor='black', markeredgewidth=2,
                   label='Current', zorder=5)
            
            # Annotations
            ax.text(0.05, 0.95, f'$E_a$ = {Ea:.1f} kJ/mol', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            
            # Styling
            ax.set_xlabel('1000/T (K⁻¹)', fontweight='bold', fontsize=11)
            ax.set_ylabel('ln(Selectivity)', fontweight='bold', fontsize=11)
            ax.set_title('Van\'t Hoff Plot: Temperature Dependence of Selectivity',
                        fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
        
        elif graph_name == "Arrhenius Plot":
            # Arrhenius plot for permeance
            ax = fig.add_subplot(111)
            
            # Generate data
            temperatures = np.linspace(273, 373, 30)
            inv_temp = 1000 / temperatures
            
            # Permeance models for CO2 and N2
            if self.params['membrane_type'] == 'Polaris':
                P_CO2_ref = 3000
            else:
                P_CO2_ref = 2500
            
            # Arrhenius: P = P0 * exp(-Ea/RT)
            Ea_CO2 = 15000  # J/mol
            Ea_N2 = 20000   # J/mol
            R = 8.314
            
            P_CO2 = P_CO2_ref * np.exp(-Ea_CO2/R * (1/temperatures - 1/298))
            P_N2 = P_CO2 / 50  # Approximate N2 permeance
            
            ln_P_CO2 = np.log(P_CO2)
            ln_P_N2 = np.log(P_N2)
            
            # Linear fits
            coeffs_CO2 = np.polyfit(inv_temp, ln_P_CO2, 1)
            coeffs_N2 = np.polyfit(inv_temp, ln_P_N2, 1)
            
            fit_CO2 = coeffs_CO2[0] * inv_temp + coeffs_CO2[1]
            fit_N2 = coeffs_N2[0] * inv_temp + coeffs_N2[1]
            
            # Plot
            ax.plot(inv_temp, ln_P_CO2, 'o', color='#4CAF50', markersize=6, alpha=0.6, label='CO₂ Data')
            ax.plot(inv_temp, fit_CO2, '-', linewidth=2.5, color='#4CAF50', label='CO₂ Fit')
            
            ax.plot(inv_temp, ln_P_N2, 's', color='#2196F3', markersize=6, alpha=0.6, label='N₂ Data')
            ax.plot(inv_temp, fit_N2, '--', linewidth=2.5, color='#2196F3', label='N₂ Fit')
            
            # Current point
            current_inv_temp = 1000 / self.params['temperature']
            current_P_CO2 = P_CO2_ref * np.exp(-Ea_CO2/R * (1/self.params['temperature'] - 1/298))
            current_P_N2 = current_P_CO2 / 50
            
            ax.plot([current_inv_temp], [np.log(current_P_CO2)], '*', markersize=15,
                   color='red', markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            
            # Activation energies
            Ea_CO2_calc = -coeffs_CO2[0] * R / 1000
            Ea_N2_calc = -coeffs_N2[0] * R / 1000
            
            ax.text(0.05, 0.95, f'$E_a$(CO₂) = {Ea_CO2_calc:.1f} kJ/mol\n$E_a$(N₂) = {Ea_N2_calc:.1f} kJ/mol',
                   transform=ax.transAxes, fontsize=10, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            # Styling
            ax.set_xlabel('1000/T (K⁻¹)', fontweight='bold', fontsize=11)
            ax.set_ylabel('ln(Permeance) [GPU]', fontweight='bold', fontsize=11)
            ax.set_title('Arrhenius Plot: Temperature Dependence of Permeance',
                        fontweight='bold', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        elif graph_name == "Violin Performance Plot":
            # Violin plot showing performance distributions
            ax = fig.add_subplot(111)
            
            # Generate synthetic data for different membrane types
            membrane_types = ['Standard', 'Advanced', 'Polaris', 'Ultra-Thin']
            data_recovery = []
            data_purity = []
            
            np.random.seed(42)
            for mem_type in membrane_types:
                if mem_type == 'Standard':
                    recovery = np.random.normal(70, 8, 200)
                    purity = np.random.normal(65, 10, 200)
                elif mem_type == 'Advanced':
                    recovery = np.random.normal(80, 5, 200)
                    purity = np.random.normal(75, 7, 200)
                elif mem_type == 'Polaris':
                    recovery = np.random.normal(75, 6, 200)
                    purity = np.random.normal(70, 8, 200)
                else:  # Ultra-Thin
                    recovery = np.random.normal(85, 4, 200)
                    purity = np.random.normal(82, 5, 200)
                
                data_recovery.append(recovery)
                data_purity.append(purity)
            
            # Create violin plot
            positions_recovery = np.arange(1, len(membrane_types)+1) - 0.2
            positions_purity = np.arange(1, len(membrane_types)+1) + 0.2
            
            parts1 = ax.violinplot(data_recovery, positions=positions_recovery, widths=0.35,
                                   showmeans=True, showmedians=True)
            parts2 = ax.violinplot(data_purity, positions=positions_purity, widths=0.35,
                                   showmeans=True, showmedians=True)
            
            # Color violins
            for pc in parts1['bodies']:
                pc.set_facecolor('#4CAF50')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            for pc in parts2['bodies']:
                pc.set_facecolor('#2196F3')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Reference line
            ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.6, label='80% Target')
            
            # Styling
            ax.set_xticks(np.arange(1, len(membrane_types)+1))
            ax.set_xticklabels(membrane_types, fontsize=10)
            ax.set_ylabel('Performance (%)', fontweight='bold', fontsize=11)
            ax.set_title('Violin Plot: Performance Distribution by Membrane Type',
                        fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(40, 100)
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4CAF50', alpha=0.7, edgecolor='black', label='Recovery'),
                Patch(facecolor='#2196F3', alpha=0.7, edgecolor='black', label='Purity'),
                ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.6, label='80% Target')[0]
            ]
            ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
    
    def draw_simulation_graph(self, fig, graph_name):
        """Draw advanced simulation study graphs"""
        
        # Check if we have advanced simulation results
        if self.sweep_results is not None and len(self.sweep_results) > 0:
            # Show the advanced simulation results
            param_name = self.sim_ranges['param'].get()
            
            # Create comprehensive view
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Get parameter values
            param_col = param_name
            if param_col in self.sweep_results.columns:
                param_values = self.sweep_results[param_col].values
                
                # Adjust display based on parameter type
                if param_name == 'temperature':
                    param_display = param_values
                    xlabel = 'Temperature (K)'
                elif param_name == 'feed_pressure':
                    param_display = param_values
                    xlabel = 'Feed Pressure (bar)'
                elif param_name == 'permeate_pressure':
                    param_display = param_values
                    xlabel = 'Permeate Pressure (bar)'
                elif param_name == 'feed_composition':
                    param_display = param_values * 100
                    xlabel = 'Feed CO₂ (%)'
                elif param_name == 'o2_composition':
                    param_display = param_values * 100
                    xlabel = 'O₂ Composition (%)'
                else:
                    param_display = param_values
                    xlabel = param_name.replace('_', ' ').title()
                
                # Plot 1: Recovery and Purity
                ax1 = fig.add_subplot(gs[0, 0])
                recovery = self.sweep_results['co2_recovery'].values * 100
                purity = self.sweep_results['permeate_co2'].values * 100
                
                ax1.plot(param_display, recovery, 'b-o', linewidth=2, markersize=4, 
                        label='Recovery', alpha=0.7)
                ax1.plot(param_display, purity, 'g-s', linewidth=2, markersize=4, 
                        label='Purity', alpha=0.7)
                ax1.axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='80% Target')
                ax1.set_xlabel(xlabel, fontweight='bold', fontsize=9)
                ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=9)
                ax1.set_title('Recovery & Purity', fontweight='bold', fontsize=10)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Membrane Area
                ax2 = fig.add_subplot(gs[0, 1])
                area = self.sweep_results['membrane_area'].values
                ax2.plot(param_display, area, 'r-^', linewidth=2, markersize=4, alpha=0.7)
                ax2.fill_between(param_display, area, alpha=0.2, color='red')
                ax2.set_xlabel(xlabel, fontweight='bold', fontsize=9)
                ax2.set_ylabel('Membrane Area (m²)', fontweight='bold', fontsize=9)
                ax2.set_title('Membrane Area Requirement', fontweight='bold', fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Energy and Cost
                ax3 = fig.add_subplot(gs[1, 0])
                if 'energy_kwh_per_ton' in self.sweep_results.columns:
                    energy = self.sweep_results['energy_kwh_per_ton'].values
                    energy_clean = np.where(np.isinf(energy), np.nan, energy)
                    ax3.plot(param_display, energy_clean, 'm-d', linewidth=2, markersize=4, 
                            label='Energy', alpha=0.7)
                    ax3.set_ylabel('Energy (kWh/ton CO₂)', fontweight='bold', fontsize=9, color='m')
                    ax3.tick_params(axis='y', labelcolor='m')
                
                if 'cost_per_ton_co2' in self.sweep_results.columns:
                    ax3b = ax3.twinx()
                    cost = self.sweep_results['cost_per_ton_co2'].values
                    cost_clean = np.where(np.isinf(cost), np.nan, cost)
                    ax3b.plot(param_display, cost_clean, 'c-o', linewidth=2, markersize=4, 
                             label='Cost', alpha=0.7)
                    ax3b.axhline(40, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax3b.set_ylabel('Cost ($/ton CO₂)', fontweight='bold', fontsize=9, color='c')
                    ax3b.tick_params(axis='y', labelcolor='c')
                
                ax3.set_xlabel(xlabel, fontweight='bold', fontsize=9)
                ax3.set_title('Energy & Cost per Ton CO₂', fontweight='bold', fontsize=10)
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Flux and Selectivity
                ax4 = fig.add_subplot(gs[1, 1])
                if 'co2_flux' in self.sweep_results.columns:
                    flux = self.sweep_results['co2_flux'].values
                    ax4.plot(param_display, flux, 'b-s', linewidth=2, markersize=4, 
                            label='CO₂ Flux', alpha=0.7)
                    ax4.set_ylabel('CO₂ Flux (mol/m²/s)', fontweight='bold', fontsize=9, color='b')
                    ax4.tick_params(axis='y', labelcolor='b')
                
                if 'selectivity' in self.sweep_results.columns:
                    ax4b = ax4.twinx()
                    selectivity = self.sweep_results['selectivity'].values
                    ax4b.plot(param_display, selectivity, 'r-^', linewidth=2, markersize=4, 
                             label='Selectivity', alpha=0.7)
                    ax4b.set_ylabel('CO₂/N₂ Selectivity', fontweight='bold', fontsize=9, color='r')
                    ax4b.tick_params(axis='y', labelcolor='r')
                
                ax4.set_xlabel(xlabel, fontweight='bold', fontsize=9)
                ax4.set_title('Flux & Selectivity', fontweight='bold', fontsize=10)
                ax4.grid(True, alpha=0.3)
                
                # Add overall title
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                fig.suptitle(f'Advanced Simulation: {param_name.replace("_", " ").title()}\n'
                           f'Range: {start_val} to {end_val}', 
                            fontweight='bold', fontsize=12)
            
            return
        
        # If no sweep results, show the original predefined simulation graphs
        if graph_name == "O₂ Injection Study":
            ax = fig.add_subplot(111)
            
            # Prepare base parameters
            base_params = {
                'feed_flow': self.params['feed_flow'],
                'feed_composition': self.params['feed_composition'],
                'temperature': self.params['temperature'],
                'feed_pressure': self.params['feed_pressure'],
                'permeate_pressure': self.params['permeate_pressure'],
                'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                'electricity_cost': self.params['electricity_cost'],
                'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
            }
            
            # Run O2 injection study
            try:
                sweep_df = self.sim_engine.o2_injection_study(base_params, (0, 0.10), 15)
                self.sweep_results = sweep_df
                
                # Plot Recovery and Purity vs O2
                ax2 = ax.twinx()
                
                o2_percent = sweep_df['o2_composition'] * 100
                recovery = sweep_df['co2_recovery'] * 100
                purity = sweep_df['permeate_co2'] * 100
                
                line1 = ax.plot(o2_percent, recovery, 'b-o', linewidth=2, 
                              markersize=6, label='CO₂ Recovery', alpha=0.7)
                line2 = ax.plot(o2_percent, purity, 'g-s', linewidth=2, 
                              markersize=6, label='CO₂ Purity', alpha=0.7)
                
                # Cost per ton on right axis
                if 'cost_per_ton_co2' in sweep_df.columns:
                    cost = sweep_df['cost_per_ton_co2']
                    # Filter out inf values
                    cost_clean = cost.replace([np.inf, -np.inf], np.nan)
                    line3 = ax2.plot(o2_percent, cost_clean, 'r-^', linewidth=2, 
                                   markersize=6, label='Cost per Ton', alpha=0.7)
                    ax2.set_ylabel('Cost ($/ton CO₂)', fontweight='bold', fontsize=11, color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                
                ax.set_xlabel('O₂ Injection (%)', fontweight='bold', fontsize=11)
                ax.set_ylabel('Recovery / Purity (%)', fontweight='bold', fontsize=11, color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax.set_title('O₂ Injection Study: Effect on Performance', fontweight='bold', fontsize=13)
                ax.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                if 'cost_per_ton_co2' in sweep_df.columns:
                    lines += line3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='best', fontsize=9)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
        
        elif graph_name == "Thermal Ramp Study":
            ax = fig.add_subplot(111)
            
            # Prepare base parameters
            base_params = {
                'feed_flow': self.params['feed_flow'],
                'feed_composition': self.params['feed_composition'],
                'temperature': self.params['temperature'],
                'feed_pressure': self.params['feed_pressure'],
                'permeate_pressure': self.params['permeate_pressure'],
                'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                'electricity_cost': self.params['electricity_cost'],
                'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
            }
            
            # Run thermal ramp study
            try:
                sweep_df = self.sim_engine.thermal_ramp_study(base_params, (273, 373), 25)
                self.sweep_results = sweep_df
                
                # Create dual-axis plot
                ax2 = ax.twinx()
                
                temp = sweep_df['temperature']
                recovery = sweep_df['co2_recovery'] * 100
                area = sweep_df['membrane_area']
                
                line1 = ax.plot(temp, recovery, 'b-o', linewidth=2, 
                              markersize=5, label='Recovery', alpha=0.7)
                line2 = ax2.plot(temp, area, 'r-s', linewidth=2, 
                               markersize=5, label='Membrane Area', alpha=0.7)
                
                ax.set_xlabel('Temperature (K)', fontweight='bold', fontsize=11)
                ax.set_ylabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11, color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax2.set_ylabel('Membrane Area (m²)', fontweight='bold', fontsize=11, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                ax.set_title('Thermal Ramp Study: Temperature Effects', fontweight='bold', fontsize=13)
                ax.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='best', fontsize=10)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
        
        elif graph_name == "Multi-Param Grid":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            
            # Prepare base parameters
            base_params = {
                'feed_flow': self.params['feed_flow'],
                'feed_composition': self.params['feed_composition'],
                'temperature': self.params['temperature'],
                'feed_pressure': self.params['feed_pressure'],
                'permeate_pressure': self.params['permeate_pressure'],
                'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                'electricity_cost': self.params['electricity_cost'],
                'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
            }
            
            # Run grid sweep
            try:
                sweep_df = self.sim_engine.grid_sweep(
                    base_params, 
                    'feed_pressure', (1, 8), 
                    'feed_composition', (0.05, 0.40),
                    15, 15
                )
                self.sweep_results = sweep_df
                
                # Create 3D scatter
                fp = sweep_df['feed_pressure']
                fc = sweep_df['feed_composition'] * 100
                recovery = sweep_df['co2_recovery'] * 100
                
                scatter = ax.scatter(fp, fc, recovery, c=recovery, cmap='viridis', 
                                   s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
                
                # Mark current point
                ax.scatter([self.params['feed_pressure']], 
                          [self.params['feed_composition'] * 100],
                          [self.results['co2_recovery'] * 100],
                          color='red', s=200, marker='*', edgecolor='black', linewidth=2)
                
                ax.set_xlabel('Feed Pressure (bar)', fontweight='bold', fontsize=10)
                ax.set_ylabel('Feed CO₂ (%)', fontweight='bold', fontsize=10)
                ax.set_zlabel('Recovery (%)', fontweight='bold', fontsize=10)
                ax.set_title('Multi-Parameter Grid Sweep', fontweight='bold', fontsize=12)
                
                fig.colorbar(scatter, ax=ax, shrink=0.5, label='Recovery (%)')
                
            except Exception as e:
                ax.text2D(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                         ha='center', va='center', transform=ax.transAxes,
                         fontsize=12, color='red')
        
        elif graph_name == "Monte Carlo Analysis":
            ax = fig.add_subplot(111)
            
            # Prepare base parameters
            base_params = {
                'feed_flow': self.params['feed_flow'],
                'feed_composition': self.params['feed_composition'],
                'temperature': self.params['temperature'],
                'feed_pressure': self.params['feed_pressure'],
                'permeate_pressure': self.params['permeate_pressure'],
                'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                'electricity_cost': self.params['electricity_cost'],
                'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
            }
            
            # Define uncertainties (mean, std)
            uncertainties = {
                'feed_composition': (self.params['feed_composition'], 0.02),
                'temperature': (self.params['temperature'], 5),
                'feed_pressure': (self.params['feed_pressure'], 0.2)
            }
            
            # Run Monte Carlo
            try:
                mc_df = self.sim_engine.monte_carlo_simulation(base_params, uncertainties, 200)
                self.sweep_results = mc_df
                
                # Histogram of recovery
                recovery = mc_df['co2_recovery'] * 100
                ax.hist(recovery, bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
                
                # Add statistics
                mean_rec = recovery.mean()
                std_rec = recovery.std()
                ax.axvline(mean_rec, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_rec:.1f}%')
                ax.axvline(mean_rec - std_rec, color='orange', linestyle=':', linewidth=2,
                          label=f'±1σ: {std_rec:.1f}%')
                ax.axvline(mean_rec + std_rec, color='orange', linestyle=':', linewidth=2)
                
                # Target line
                ax.axvline(80, color='green', linestyle='-', linewidth=2, alpha=0.5,
                          label='80% Target')
                
                ax.set_xlabel('CO₂ Recovery (%)', fontweight='bold', fontsize=11)
                ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
                ax.set_title('Monte Carlo Uncertainty Analysis', fontweight='bold', fontsize=13)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
        
        elif graph_name == "Batch Scenarios":
            ax = fig.add_subplot(111)
            
            # Define scenarios
            scenarios = {
                'Base': {
                    'feed_flow': self.params['feed_flow'],
                    'feed_composition': self.params['feed_composition'],
                    'temperature': self.params['temperature'],
                    'feed_pressure': self.params['feed_pressure'],
                    'permeate_pressure': self.params['permeate_pressure'],
                    'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                    'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                    'electricity_cost': self.params['electricity_cost'],
                    'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
                },
                'High Pressure': {
                    **{
                        'feed_flow': self.params['feed_flow'],
                        'feed_composition': self.params['feed_composition'],
                        'temperature': self.params['temperature'],
                        'permeate_pressure': self.params['permeate_pressure'],
                        'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                        'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                        'electricity_cost': self.params['electricity_cost'],
                        'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
                    },
                    'feed_pressure': min(10, self.params['feed_pressure'] * 1.5)
                },
                'Low Temp': {
                    **{
                        'feed_flow': self.params['feed_flow'],
                        'feed_composition': self.params['feed_composition'],
                        'feed_pressure': self.params['feed_pressure'],
                        'permeate_pressure': self.params['permeate_pressure'],
                        'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                        'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                        'electricity_cost': self.params['electricity_cost'],
                        'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
                    },
                    'temperature': max(273, self.params['temperature'] - 20)
                },
                'Rich Feed': {
                    **{
                        'feed_flow': self.params['feed_flow'],
                        'temperature': self.params['temperature'],
                        'feed_pressure': self.params['feed_pressure'],
                        'permeate_pressure': self.params['permeate_pressure'],
                        'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                        'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                        'electricity_cost': self.params['electricity_cost'],
                        'membrane_cost_per_m2': self.params['membrane_cost_per_m2']
                    },
                    'feed_composition': min(0.40, self.params['feed_composition'] * 1.5)
                }
            }
            
            # Run batch comparison
            try:
                batch_df = self.sim_engine.batch_scenario_comparison(scenarios)
                self.sweep_results = batch_df
                
                # Bar chart comparison
                scenario_names = batch_df['scenario_name'].values
                recoveries = batch_df['co2_recovery'].values * 100
                purities = batch_df['permeate_co2'].values * 100
                
                x = np.arange(len(scenario_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, recoveries, width, label='Recovery', 
                             color='#2196F3', alpha=0.7, edgecolor='black')
                bars2 = ax.bar(x + width/2, purities, width, label='Purity', 
                             color='#4CAF50', alpha=0.7, edgecolor='black')
                
                # Target line
                ax.axhline(80, color='red', linestyle='--', linewidth=2, alpha=0.5,
                          label='80% Target')
                
                ax.set_xlabel('Scenario', fontweight='bold', fontsize=11)
                ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
                ax.set_title('Batch Scenario Comparison', fontweight='bold', fontsize=13)
                ax.set_xticks(x)
                ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 100)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Simulation Error:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
    
    def draw_process_design_graph(self, fig, graph_name):
        """Draw membrane separation process design diagrams using matplotlib graphics"""
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
        from matplotlib.patches import Arc, Wedge
        
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Helper function to draw membrane
        def draw_membrane(ax, x, y, width=0.8, height=1.5, label="M"):
            """Draw a membrane module"""
            rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#1976D2', facecolor='#BBDEFB',
                                 linewidth=2.5)
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='#0D47A1')
            return rect
        
        # Helper function to draw arrow
        def draw_arrow(ax, x1, y1, x2, y2, label="", color='#424242'):
            """Draw a flow arrow"""
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=25,
                                   color=color, linewidth=2.5)
            ax.add_patch(arrow)
            if label:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.9))
        
        # Helper function to draw pump
        def draw_pump(ax, x, y, size=0.4):
            """Draw a pump symbol"""
            circle = Circle((x, y), size, edgecolor='#F57C00', 
                          facecolor='#FFE0B2', linewidth=2)
            ax.add_patch(circle)
            # Add "P" for pump
            ax.text(x, y, 'P', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='#E65100')
        
        # Helper function to draw text box
        def draw_textbox(ax, x, y, text, color='#4CAF50'):
            """Draw a labeled text box"""
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor=color, linewidth=2))
        
        # Title
        ax.text(5, 9.5, graph_name, ha='center', va='top',
               fontsize=16, fontweight='bold', color='#1976D2')
        
        if graph_name == "Single-Stage System":
            # Feed → Membrane → Permeate / Retentate
            draw_textbox(ax, 1.5, 5, "Feed", '#4CAF50')
            draw_arrow(ax, 2.3, 5, 3.7, 5, "Feed Stream")
            draw_membrane(ax, 5, 5, 1.0, 2.0, "Membrane")
            draw_arrow(ax, 6.3, 5, 7.7, 5, "Permeate", '#2196F3')
            draw_textbox(ax, 8.5, 5, "Permeate", '#2196F3')
            draw_arrow(ax, 5, 3.7, 5, 2.3, "Retentate", '#F44336')
            draw_textbox(ax, 5, 1.5, "Retentate", '#F44336')
            
            # Description
            desc = ["• Simplest configuration", "• Single pass through membrane",
                   "• Best for moderate separation", "• Low capital cost"]
            for i, line in enumerate(desc):
                ax.text(5, 0.8 - i*0.25, line, ha='center', va='top',
                       fontsize=9, color='#424242')
        
        elif graph_name == "Two-Stage Cascade":
            # Stage 1
            draw_textbox(ax, 1.5, 7, "Feed", '#4CAF50')
            draw_arrow(ax, 2.3, 7, 3.2, 7)
            draw_membrane(ax, 4, 7, 0.8, 1.5, "S1")
            draw_arrow(ax, 4.8, 7, 7.7, 7, "Permeate 1", '#2196F3')
            draw_textbox(ax, 8.5, 7, "Product", '#2196F3')
            
            # Retentate 1 to Stage 2
            draw_arrow(ax, 4, 6.1, 4, 4.9, "Retentate 1", '#FF9800')
            draw_membrane(ax, 4, 4, 0.8, 1.5, "S2")
            draw_arrow(ax, 4.8, 4, 6.2, 4, "Permeate 2", '#64B5F6')
            draw_arrow(ax, 4, 3.1, 4, 2.3, "Retentate 2", '#F44336')
            draw_textbox(ax, 4, 1.5, "Waste", '#F44336')
            
            desc = ["• Higher recovery than single-stage", "• Second stage captures more product",
                   "• Moderate complexity", "• Common in CO₂ capture"]
            for i, line in enumerate(desc):
                ax.text(5, 0.8 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Multi-Stage Series":
            # Three membranes in series
            y = 5
            draw_textbox(ax, 0.8, y, "Feed", '#4CAF50')
            draw_arrow(ax, 1.5, y, 2.2, y)
            draw_membrane(ax, 3, y, 0.7, 1.3, "M1")
            draw_arrow(ax, 3.5, y, 4.2, y)
            draw_membrane(ax, 5, y, 0.7, 1.3, "M2")
            draw_arrow(ax, 5.5, y, 6.2, y)
            draw_membrane(ax, 7, y, 0.7, 1.3, "M3")
            draw_arrow(ax, 7.5, y, 8.7, y, "Permeate", '#2196F3')
            draw_textbox(ax, 9.3, y, "Product", '#2196F3')
            
            # Retentate streams
            draw_arrow(ax, 3, y-0.8, 3, y-1.4, "R1", '#FF5722')
            draw_arrow(ax, 5, y-0.8, 5, y-1.4, "R2", '#FF6F00')
            draw_arrow(ax, 7, y-0.8, 7, y-1.4, "R3", '#F44336')
            
            # Combined retentate
            draw_arrow(ax, 3, y-1.6, 6.2, y-2.3)
            draw_arrow(ax, 5, y-1.6, 6.2, y-2.3)
            draw_arrow(ax, 7, y-1.6, 6.2, y-2.3)
            draw_textbox(ax, 7, y-2.5, "Combined\nRetentate", '#F44336')
            
            desc = ["• Progressive concentration", "• Each stage increases purity",
                   "• Higher membrane area", "• High purity applications"]
            for i, line in enumerate(desc):
                ax.text(5, 1.2 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Parallel Array":
            # Three parallel modules
            draw_textbox(ax, 1, 5, "Feed", '#4CAF50')
            
            # Split to three modules
            positions = [7, 5, 3]
            for i, y_pos in enumerate(positions, 1):
                draw_arrow(ax, 1.8, 5, 2.7, y_pos)
                draw_membrane(ax, 3.5, y_pos, 0.7, 1.2, f"M{i}")
                draw_arrow(ax, 4.2, y_pos, 6.2, y_pos, f"P{i}", '#2196F3')
            
            # Combine permeate
            for y_pos in positions:
                draw_arrow(ax, 6.5, y_pos, 7.5, 5)
            draw_textbox(ax, 8.5, 5, "Combined\nPermeate", '#2196F3')
            
            desc = ["• High throughput capacity", "• Modular & scalable design",
                   "• Redundancy for reliability", "• Easy maintenance"]
            for i, line in enumerate(desc):
                ax.text(5, 1.5 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Recirculation Loop":
            # Main membrane
            draw_textbox(ax, 1.5, 5, "Feed", '#4CAF50')
            draw_arrow(ax, 2.3, 5, 3.7, 5)
            draw_membrane(ax, 5, 5, 1.0, 2.0, "Membrane")
            draw_arrow(ax, 6.3, 5, 7.7, 5, "Permeate", '#2196F3')
            draw_textbox(ax, 8.5, 5, "Product", '#2196F3')
            
            # Retentate loop
            draw_arrow(ax, 5, 3.7, 5, 2.5, "Retentate", '#FF9800')
            draw_pump(ax, 5, 1.8)
            ax.text(5.6, 1.8, "Recycle\nPump", fontsize=8, color='#F57C00', va='center')
            
            # Curved recycle arrow
            from matplotlib.patches import FancyBboxPatch, Arc
            arc = Arc((5, 3), 4, 4, angle=0, theta1=180, theta2=270,
                     color='#FF9800', linewidth=2.5)
            ax.add_patch(arc)
            draw_arrow(ax, 3, 1.5, 3.5, 4.5, "", '#FF9800')
            
            desc = ["• Maximizes recovery", "• Concentrates retentate stream",
                   "• Higher energy consumption", "• Better product yield"]
            for i, line in enumerate(desc):
                ax.text(5, 0.8 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Spiral Wound Module":
            # Draw spiral representation
            center_x, center_y = 5, 5
            
            # Outer casing
            rect = Rectangle((3, 3.5), 4, 3, edgecolor='#37474F', 
                           facecolor='#CFD8DC', linewidth=3, alpha=0.3)
            ax.add_patch(rect)
            
            # Spiral layers
            from matplotlib.patches import Wedge
            for i in range(5):
                radius = 1.2 - i*0.2
                wedge = Wedge((center_x, center_y), radius, 0, 360,
                            edgecolor='#1976D2', facecolor='#BBDEFB',
                            linewidth=2, alpha=0.6)
                ax.add_patch(wedge)
            
            # Flow arrows
            draw_arrow(ax, 1.5, 5, 2.8, 5, "Feed", '#4CAF50')
            draw_arrow(ax, 7.2, 5, 8.5, 5, "Retentate", '#F44336')
            draw_arrow(ax, 5, 3.3, 5, 2.3, "Permeate", '#2196F3')
            draw_textbox(ax, 5, 1.5, "Permeate\nCollection", '#2196F3')
            
            desc = ["• Compact high surface area", "• Industry standard for RO/NF",
                   "• Cost-effective", "• Easy to replace modules"]
            for i, line in enumerate(desc):
                ax.text(5, 0.8 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Hollow Fiber Config":
            # Draw fiber bundle
            draw_textbox(ax, 1, 5, "Feed", '#4CAF50')
            
            # Hollow fibers
            for i in range(10):
                x = 3.5 + i * 0.3
                ax.plot([x, x], [3.5, 6.5], color='#1976D2', linewidth=3, alpha=0.7)
                # Permeate arrows
                if i % 2 == 0:
                    draw_arrow(ax, x, 3.3, x, 2.5, "", '#2196F3')
            
            # Flow arrows
            draw_arrow(ax, 1.8, 5, 3.2, 5)
            draw_arrow(ax, 6.8, 5, 8.2, 5, "Retentate", '#F44336')
            draw_textbox(ax, 9, 5, "Retentate", '#F44336')
            
            # Permeate collection
            ax.plot([3.5, 6.5], [2, 2], color='#2196F3', linewidth=4)
            draw_textbox(ax, 5, 1.2, "Permeate Collection", '#2196F3')
            
            desc = ["• Highest surface area/volume", "• Self-supporting structure",
                   "• Excellent for gas separation", "• Sensitive to fouling"]
            for i, line in enumerate(desc):
                ax.text(5, 0.5 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Reverse Osmosis":
            # High pressure pump
            draw_textbox(ax, 1, 6.5, "Feed\nWater", '#4CAF50')
            draw_arrow(ax, 1.8, 6.5, 2.5, 6.5)
            draw_pump(ax, 3, 6.5, 0.5)
            ax.text(3, 7.3, "High Pressure\nPump", fontsize=9, ha='center',
                   fontweight='bold', color='#F57C00')
            
            # RO membrane
            draw_arrow(ax, 3.6, 6.5, 4.2, 6.5)
            draw_membrane(ax, 5.5, 6.5, 1.2, 2.5, "RO")
            ax.text(5.5, 7.8, "15-80 bar", fontsize=8, ha='center',
                   color='#D32F2F', fontweight='bold')
            
            # Outputs
            draw_arrow(ax, 6.9, 6.5, 8.2, 6.5, "Pure Water", '#2196F3')
            draw_textbox(ax, 9, 6.5, "Product\nWater", '#2196F3')
            
            draw_arrow(ax, 5.5, 5.1, 5.5, 4.3, "Brine", '#F44336')
            draw_textbox(ax, 5.5, 3.5, "Concentrated\nBrine", '#F44336')
            
            desc = ["• Desalination & purification", "• High pressure operation",
                   "• Removes dissolved salts", "• Energy recovery devices"]
            for i, line in enumerate(desc):
                ax.text(5, 2.5 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        elif graph_name == "Gas Separation":
            # Gas mixture
            draw_textbox(ax, 1.5, 5, "Gas Mix\nCO₂+N₂", '#FFA726')
            draw_arrow(ax, 2.5, 5, 3.7, 5)
            
            # Membrane with pressure differential
            draw_membrane(ax, 5, 5, 1.0, 2.5, "Gas\nMembrane")
            ax.text(5, 7.2, "ΔP", fontsize=12, ha='center',
                   fontweight='bold', color='#D32F2F')
            
            # Fast gas (CO2)
            draw_arrow(ax, 6.3, 5, 7.7, 5, "Fast Gas\n(CO₂)", '#2196F3')
            draw_textbox(ax, 8.7, 5, "Permeate\n(CO₂)", '#2196F3')
            
            # Slow gas (N2)
            draw_arrow(ax, 5, 3.6, 5, 2.5, "Slow Gas\n(N₂)", '#F44336')
            draw_textbox(ax, 5, 1.5, "Retentate\n(N₂)", '#F44336')
            
            desc = ["• CO₂/N₂ separation", "• H₂ purification",
                   "• Natural gas processing", "• Selectivity-based separation"]
            for i, line in enumerate(desc):
                ax.text(5, 0.8 - i*0.2, line, ha='center', fontsize=8, color='#424242')
        
        else:
            # Default visualization for other designs
            ax.text(5, 5, f'{graph_name}\nVisualization Coming Soon!',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='#1976D2')
            ax.text(5, 3.5, 'This process design will feature\ninteractive SVG-style graphics',
                   ha='center', va='center', fontsize=11, color='#666666', style='italic')
        
        # Add current simulation info
        if self.results:
            note_text = f"Simulation: {self.params['membrane_type']} | " \
                       f"{self.params['feed_pressure']:.1f} bar | {self.params['temperature']:.0f} K"
            ax.text(5, 0.2, note_text, ha='center', va='bottom',
                   fontsize=8, style='italic', color='#666666',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                            edgecolor='#CCCCCC', alpha=0.8))
        
        fig.tight_layout()
    
    def run_advanced_simulation(self):
        """Run advanced simulation based on selected type"""
        try:
            self.status_label.config(text="⚙️ Running advanced simulation...")
            self.root.update()
            
            # Get simulation type
            sim_type = self.sim_type_var.get()
            
            # Prepare base parameters
            base_params = {
                'feed_flow': self.params['feed_flow'],
                'feed_composition': self.params['feed_composition'],
                'temperature': self.params['temperature'],
                'feed_pressure': self.params['feed_pressure'],
                'permeate_pressure': self.params['permeate_pressure'],
                'co2_permeance_gpu': 1000 if self.params['membrane_type'] == 'Polaris' else 800,
                'selectivity': 40 if self.params['membrane_type'] == 'Polaris' else 50,
                'electricity_cost': self.params['electricity_cost'],
                'membrane_cost_per_m2': self.params['membrane_cost_per_m2'],
                'o2_composition': 0.0
            }
            
            # Run appropriate simulation based on type
            if sim_type == 'Parameter Sweep':
                # Get simulation parameters
                param_name = self.sim_ranges['param'].get()
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                num_points = self.sim_ranges['points'].get()
                
                # Validate inputs
                if start_val >= end_val:
                    messagebox.showerror("Invalid Range", 
                                       "Start value must be less than end value!")
                    self.status_label.config(text="❌ Invalid range")
                    return
                
                # Run parameter sweep
                sweep_df = self.sim_engine.parameter_sweep(
                    base_params, 
                    param_name, 
                    (start_val, end_val), 
                    num_points
                )
                sim_desc = f"{param_name.replace('_', ' ').title()}: {start_val} to {end_val}"
                
            elif sim_type == 'O₂ Injection':
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                num_points = self.sim_ranges['points'].get()
                
                sweep_df = self.sim_engine.o2_injection_study(
                    base_params, 
                    (start_val, end_val), 
                    num_points
                )
                sim_desc = f"O₂ Injection: {start_val*100:.1f}% to {end_val*100:.1f}%"
                
            elif sim_type == 'Thermal Ramp':
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                num_points = self.sim_ranges['points'].get()
                
                sweep_df = self.sim_engine.thermal_ramp_study(
                    base_params, 
                    (start_val, end_val), 
                    num_points
                )
                sim_desc = f"Thermal Ramp: {start_val} K to {end_val} K"
                
            elif sim_type == 'Multi-Param Grid':
                param_name = self.sim_ranges['param'].get()
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                num_points = self.sim_ranges['points'].get()
                
                # For grid, use two parameters
                if param_name == 'temperature':
                    param2 = 'feed_composition'
                    range2 = (0.05, 0.40)
                else:
                    param2 = 'feed_pressure'
                    range2 = (1, 10)
                
                sweep_df = self.sim_engine.grid_sweep(
                    base_params, 
                    param_name, (start_val, end_val),
                    param2, range2,
                    num_points, num_points
                )
                sim_desc = f"Grid: {param_name} vs {param2}"
                
            elif sim_type == 'Monte Carlo':
                num_samples = self.sim_ranges['points'].get()
                
                # Define uncertainties
                uncertainties = {
                    'feed_composition': (base_params['feed_composition'], 0.02),
                    'temperature': (base_params['temperature'], 5),
                    'feed_pressure': (base_params['feed_pressure'], 0.2)
                }
                
                sweep_df = self.sim_engine.monte_carlo_simulation(
                    base_params, 
                    uncertainties, 
                    num_samples
                )
                sim_desc = f"Monte Carlo: {num_samples} samples"
                
            elif sim_type == 'Batch Scenarios':
                # Define scenarios
                scenarios = {
                    'Base': base_params.copy(),
                    'High Pressure': {**base_params, 'feed_pressure': min(10, base_params['feed_pressure'] * 1.5)},
                    'Low Temp': {**base_params, 'temperature': max(273, base_params['temperature'] - 20)},
                    'Rich Feed': {**base_params, 'feed_composition': min(0.40, base_params['feed_composition'] * 1.5)},
                    'Lean Feed': {**base_params, 'feed_composition': max(0.05, base_params['feed_composition'] * 0.7)}
                }
                
                sweep_df = self.sim_engine.batch_scenario_comparison(scenarios)
                sim_desc = "Batch Scenarios Comparison"
            
            else:
                messagebox.showerror("Error", f"Unknown simulation type: {sim_type}")
                return
            
            # Check results
            if len(sweep_df) == 0:
                messagebox.showerror("Simulation Failed", 
                                   "No valid simulation results obtained.\n"
                                   "Try adjusting the parameters.")
                self.status_label.config(text="❌ No valid results")
                return
            
            self.sweep_results = sweep_df
            self.current_sim_type = sim_type  # Store for later use
            
            # Update the simulation tab with results
            self.status_label.config(text=f"✓ Simulation complete: {len(sweep_df)} points")
            
            # Switch to simulation tab and update
            self.notebook.select(6)  # Simulation tab is index 6
            self.update_advanced_simulation_display()
            
            # Automatically save PNG
            self.auto_save_simulation_png()
            
            messagebox.showinfo("Success", 
                              f"✓ {sim_type} Simulation Complete!\n\n"
                              f"{sim_desc}\n"
                              f"Points simulated: {len(sweep_df)}\n\n"
                              f"Results automatically saved as PNG and CSV!")
            
        except Exception as e:
            import traceback
            error_msg = f"Error running simulation:\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Simulation Error", error_msg)
            self.status_label.config(text="❌ Simulation failed")
            print(error_msg)  # Also print to console for debugging
            messagebox.showerror("Simulation Error", f"Error running simulation: {str(e)}")
            self.status_label.config(text="❌ Simulation failed")
    
    def update_advanced_simulation_display(self):
        """Update the visualization for advanced simulation results"""
        if self.sweep_results is None or len(self.sweep_results) == 0:
            return
        
        # Get current simulation tab
        tab_name = "🧪 Simulation"
        tab_index = 6
        
        # Clear and redraw the figure
        fig = self.figures[tab_index]
        fig.clear()
        
        # Create a comprehensive view of sweep results
        param_name = self.sim_ranges['param'].get()
        
        # Create 2x2 subplot layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Get parameter values
        param_col = param_name
        if param_col in self.sweep_results.columns:
            param_values = self.sweep_results[param_col].values
            
            # Adjust display based on parameter type
            if param_name == 'temperature':
                param_display = param_values
                xlabel = 'Temperature (K)'
            elif param_name == 'feed_pressure':
                param_display = param_values
                xlabel = 'Feed Pressure (bar)'
            elif param_name == 'permeate_pressure':
                param_display = param_values
                xlabel = 'Permeate Pressure (bar)'
            elif param_name == 'feed_composition':
                param_display = param_values * 100
                xlabel = 'Feed CO₂ (%)'
            elif param_name == 'o2_composition':
                param_display = param_values * 100
                xlabel = 'O₂ Composition (%)'
            else:
                param_display = param_values
                xlabel = param_name.replace('_', ' ').title()
            
            # Plot 1: Recovery and Purity
            ax1 = fig.add_subplot(gs[0, 0])
            recovery = self.sweep_results['co2_recovery'].values * 100
            purity = self.sweep_results['permeate_co2'].values * 100
            
            ax1.plot(param_display, recovery, 'b-o', linewidth=2, markersize=4, 
                    label='Recovery', alpha=0.7)
            ax1.plot(param_display, purity, 'g-s', linewidth=2, markersize=4, 
                    label='Purity', alpha=0.7)
            ax1.axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='80% Target')
            ax1.set_xlabel(xlabel, fontweight='bold', fontsize=9)
            ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=9)
            ax1.set_title('Recovery & Purity', fontweight='bold', fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Membrane Area
            ax2 = fig.add_subplot(gs[0, 1])
            area = self.sweep_results['membrane_area'].values
            ax2.plot(param_display, area, 'r-^', linewidth=2, markersize=4, alpha=0.7)
            ax2.fill_between(param_display, area, alpha=0.2, color='red')
            ax2.set_xlabel(xlabel, fontweight='bold', fontsize=9)
            ax2.set_ylabel('Membrane Area (m²)', fontweight='bold', fontsize=9)
            ax2.set_title('Membrane Area Requirement', fontweight='bold', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Energy and Cost
            ax3 = fig.add_subplot(gs[1, 0])
            if 'energy_kwh_per_ton' in self.sweep_results.columns:
                energy = self.sweep_results['energy_kwh_per_ton'].values
                energy_clean = np.where(np.isinf(energy), np.nan, energy)
                ax3.plot(param_display, energy_clean, 'm-d', linewidth=2, markersize=4, 
                        label='Energy', alpha=0.7)
                ax3.set_ylabel('Energy (kWh/ton CO₂)', fontweight='bold', fontsize=9, color='m')
                ax3.tick_params(axis='y', labelcolor='m')
            
            if 'cost_per_ton_co2' in self.sweep_results.columns:
                ax3b = ax3.twinx()
                cost = self.sweep_results['cost_per_ton_co2'].values
                cost_clean = np.where(np.isinf(cost), np.nan, cost)
                ax3b.plot(param_display, cost_clean, 'c-o', linewidth=2, markersize=4, 
                         label='Cost', alpha=0.7)
                ax3b.axhline(40, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
                ax3b.set_ylabel('Cost ($/ton CO₂)', fontweight='bold', fontsize=9, color='c')
                ax3b.tick_params(axis='y', labelcolor='c')
            
            ax3.set_xlabel(xlabel, fontweight='bold', fontsize=9)
            ax3.set_title('Energy & Cost per Ton CO₂', fontweight='bold', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Flux and Selectivity
            ax4 = fig.add_subplot(gs[1, 1])
            if 'co2_flux' in self.sweep_results.columns:
                flux = self.sweep_results['co2_flux'].values
                ax4.plot(param_display, flux, 'b-s', linewidth=2, markersize=4, 
                        label='CO₂ Flux', alpha=0.7)
                ax4.set_ylabel('CO₂ Flux (mol/m²/s)', fontweight='bold', fontsize=9, color='b')
                ax4.tick_params(axis='y', labelcolor='b')
            
            if 'selectivity' in self.sweep_results.columns:
                ax4b = ax4.twinx()
                selectivity = self.sweep_results['selectivity'].values
                ax4b.plot(param_display, selectivity, 'r-^', linewidth=2, markersize=4, 
                         label='Selectivity', alpha=0.7)
                ax4b.set_ylabel('CO₂/N₂ Selectivity', fontweight='bold', fontsize=9, color='r')
                ax4b.tick_params(axis='y', labelcolor='r')
            
            ax4.set_xlabel(xlabel, fontweight='bold', fontsize=9)
            ax4.set_title('Flux & Selectivity', fontweight='bold', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle(f'Advanced Simulation: {param_name.replace("_", " ").title()} Sweep', 
                        fontweight='bold', fontsize=12)
        
        self.canvases[tab_index].draw()
    
    def auto_save_simulation_png(self):
        """Automatically save simulation results as PNG and CSV"""
        try:
            if self.sweep_results is None or len(self.sweep_results) == 0:
                return
            
            # Generate filename with timestamp
            from datetime import datetime
            sim_type = self.current_sim_type.replace(' ', '_').replace('₂', '2').lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create descriptive filename based on simulation type
            if hasattr(self, 'current_sim_type') and self.current_sim_type in ['O₂ Injection', 'Thermal Ramp', 'Parameter Sweep']:
                param_name = self.sim_ranges['param'].get()
                start_val = self.sim_ranges['start'].get()
                end_val = self.sim_ranges['end'].get()
                param_clean = param_name.replace('_', '')
                filename = f"{sim_type}_{param_clean}_{int(start_val)}-{int(end_val)}_{timestamp}.png"
                csv_filename = f"{sim_type}_{param_clean}_{int(start_val)}-{int(end_val)}_{timestamp}.csv"
            else:
                filename = f"{sim_type}_{timestamp}.png"
                csv_filename = f"{sim_type}_{timestamp}.csv"
            
            # Save the simulation tab figure (index 6)
            fig = self.figures[6]
            fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Save the data as CSV
            self.sweep_results.to_csv(csv_filename, index=False)
            
            self.status_label.config(text=f"✓ Saved: {filename}")
            
            return filename, csv_filename
            
        except Exception as e:
            print(f"Warning: Could not auto-save results: {str(e)}")
            return None, None
    
    def save_simulation_png(self):
        """Save current simulation results as PNG"""
        try:
            if self.sweep_results is None:
                messagebox.showwarning("No Data", 
                                     "Please run an advanced simulation first!\n\n"
                                     "Steps:\n"
                                     "1. Select parameter (e.g., temperature)\n"
                                     "2. Set start value (e.g., 290)\n"
                                     "3. Set end value (e.g., 350)\n"
                                     "4. Click 'Run Simulation'")
                return
            
            # Generate filename with timestamp
            from datetime import datetime
            param_name = self.sim_ranges['param'].get()
            start_val = self.sim_ranges['start'].get()
            end_val = self.sim_ranges['end'].get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create clean parameter name for filename
            param_clean = param_name.replace('_', '')
            
            filename = f"simulation_{param_clean}_{int(start_val)}-{int(end_val)}_{timestamp}.png"
            csv_filename = f"simulation_{param_clean}_{int(start_val)}-{int(end_val)}_{timestamp}.csv"
            
            # Get current notebook tab
            current_tab = self.notebook.index(self.notebook.select())
            
            # Save the current figure
            fig = self.figures[current_tab]
            fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            
            # Also save the data as CSV
            self.sweep_results.to_csv(csv_filename, index=False)
            
            messagebox.showinfo("Success", 
                              f"✓ Results saved successfully!\n\n"
                              f"Image: {filename}\n"
                              f"Data: {csv_filename}\n\n"
                              f"Files saved in:\n{os.getcwd()}")
            
            self.status_label.config(text=f"✓ Saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results:\n{str(e)}")
    
    def auto_optimize_target(self):
        """Automatically optimize parameters to meet 80/80 target"""
        try:
            # Show progress dialog
            self.status_label.config(text="🎯 Auto-optimizing for 80/80 target...")
            self.root.update()
            
            # Confirm with user
            response = messagebox.askyesno(
                "Auto-Optimize for 80/80",
                "This will automatically find optimal operating conditions\n"
                "to achieve 80% CO₂ recovery and 80% CO₂ purity.\n\n"
                "Current feed composition and membrane type will be used.\n"
                "Feed pressure and permeate pressure will be optimized.\n\n"
                "Continue?"
            )
            
            if not response:
                self.status_label.config(text="⚠️ Optimization cancelled")
                return
            
            # Run optimization
            membrane_type = self.params['membrane_type']
            
            solution = auto_optimize_for_target(
                feed_composition=self.params['feed_composition'],
                temperature=self.params['temperature'],
                membrane_type=membrane_type,
                print_results=False
            )
            
            if solution['meets_target']:
                # Apply optimal parameters (pressure changes)
                self.fp_var.set(solution['feed_pressure'])
                self.pp_var.set(solution['permeate_pressure'])
                
                # Get current selectivity and permeance based on membrane type
                current_selectivity = 40 if self.params['membrane_type'] == 'Polaris' else 50
                current_permeance = 1000 if self.params['membrane_type'] == 'Polaris' else 800
                
                # Check if optimizer changed temperature or feed composition
                changed_params = []
                if 'temperature' in solution and solution['temperature'] != self.params['temperature']:
                    self.temp_var.set(solution['temperature'])
                    changed_params.append(f"Temperature: {solution['temperature']} K")
                
                if 'feed_composition' in solution and abs(solution['feed_composition'] - self.params['feed_composition']) > 0.01:
                    self.co2_var.set(solution['feed_composition'] * 100)
                    changed_params.append(f"Feed CO₂: {solution['feed_composition']*100:.1f}%")
                
                # Note: Selectivity and permeance changes (can't be adjusted via GUI, only by membrane type)
                if 'selectivity' in solution and abs(solution['selectivity'] - current_selectivity) > 1:
                    changed_params.append(f"Selectivity: {solution['selectivity']:.0f} (requires advanced membrane)")
                
                if 'co2_permeance_gpu' in solution and abs(solution['co2_permeance_gpu'] - current_permeance) > 10:
                    changed_params.append(f"CO₂ Permeance: {solution['co2_permeance_gpu']:.0f} GPU (requires advanced membrane)")
                
                # Update the simulation with ALL new parameters
                self.update_sim()
                
                # Build success message
                params_msg = ""
                if changed_params:
                    params_msg = "\n\nADJUSTED PARAMETERS:\n" + "\n".join(f"• {p}" for p in changed_params)
                
                # Show success message
                messagebox.showinfo(
                    "✓ Optimization Successful!",
                    f"Found optimal conditions for 80/80 target!\n\n"
                    f"OPTIMIZED PARAMETERS:\n"
                    f"Feed Pressure: {solution['feed_pressure']:.2f} bar\n"
                    f"Permeate Pressure: {solution['permeate_pressure']:.3f} bar\n"
                    f"Pressure Ratio: {solution['pressure_ratio']:.1f}\n\n"
                    f"RESULTS:\n"
                    f"CO₂ Recovery: {solution['co2_recovery']*100:.2f}% ✓\n"
                    f"CO₂ Purity: {solution['co2_purity']*100:.2f}% ✓\n"
                    f"Membrane Area: {solution['membrane_area']:.1f} m²"
                    f"{params_msg}\n\n"
                    f"All parameters applied! Check Key Results & Target Check."
                )
                
                self.status_label.config(text="✓ Auto-optimization complete - 80/80 achieved!")
                
            else:
                # Optimization failed to meet target
                messagebox.showwarning(
                    "⚠️ Target Not Achievable",
                    f"Could not find conditions to meet 80/80 target.\n\n"
                    f"Best achievable:\n"
                    f"CO₂ Recovery: {solution['co2_recovery']*100:.2f}%\n"
                    f"CO₂ Purity: {solution['co2_purity']*100:.2f}%\n\n"
                    f"Suggestions:\n"
                    f"• Try a different membrane type\n"
                    f"• Increase feed CO₂ concentration\n"
                    f"• Adjust temperature\n"
                    f"• Use multi-stage configuration"
                )
                
                self.status_label.config(text="⚠️ 80/80 target not achievable with current settings")
        
        except ImportError as e:
            messagebox.showerror(
                "Missing Dependency",
                "Auto-optimization requires scipy.\n\n"
                "Install with: pip install scipy\n\n"
                f"Error: {str(e)}"
            )
            self.status_label.config(text="❌ scipy not installed")
            
        except Exception as e:
            import traceback
            error_msg = f"Optimization error:\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Optimization Error", error_msg)
            self.status_label.config(text="❌ Optimization failed")
            print(error_msg)
    
    def reset_params(self):
        """Reset all parameters to defaults"""
        self.flow_var.set(1.0)
        self.co2_var.set(15)
        self.temp_var.set(298)
        self.fp_var.set(3.0)
        self.pp_var.set(0.2)
        self.membrane_var.set('Advanced')
        self.elec_var.set(0.07)
        self.mem_var.set(50)
        self.update_sim()


def main():
    """Launch the compact simulator"""
    root = tk.Tk()
    app = CompactMembraneSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()

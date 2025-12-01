# Membrane CO₂ Separation Simulator

A user-friendly tool for simulating membrane-based CO₂ capture from flue gas.

## What It Does

This simulator helps you design and analyze membrane systems for capturing CO₂ from post-combustion flue gas. It calculates:
- CO₂ capture efficiency
- Required membrane area
- Energy consumption
- Operating costs (OPEX)

## Quick Start

1. **Run the simulator**
   ```bash
   python interactive_simulator_compact.py
   ```

2. **Set your parameters** in the GUI:
   - Feed flow rate
   - CO₂ concentration
   - Operating pressure and temperature
   - Membrane type

3. **Click "Run Simulation"** to see results

## Files

- `interactive_simulator_compact.py` - Main GUI application
- `membrane_separation.py` - Core membrane model
- `simulation_core.py` - Simulation engine
- `opex_calculator.py` - Cost calculations
- `auto_optimizer.py` - Automatic optimization tools

## Key Features

- **Interactive GUI** - Easy-to-use interface
- **Real-time Results** - Instant calculation and visualization
- **Cost Analysis** - OPEX calculations included
- **Parameter Sweep** - Explore different operating conditions
- **Auto Optimizer** - Find optimal settings automatically

## Requirements

- Python 3.x
- numpy
- matplotlib
- scipy
- tkinter (usually included with Python)

## Installation

```bash
# Install required packages
pip install numpy matplotlib scipy
```

## Example Use Cases

- Design membrane systems for power plant flue gas
- Optimize operating conditions
- Compare different membrane materials
- Calculate capture costs

## Building Executable

To create a standalone executable that doesn't require Python:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name=MembraneSimulator --exclude-module=PyQt5 --exclude-module=PyQt6 interactive_simulator_compact.py
```

The executable will be in the `dist` folder.

**Note for macOS users:** The build creates `MembraneSimulator.app` which you can double-click to run.

**Note for Windows users:** Use the same command - it will create `MembraneSimulator.exe`

## Support

For questions or issues, please refer to the code documentation or modify parameters in the GUI.

---
*Compact Membrane Separation Package - Updated December 2025*

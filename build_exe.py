"""
Build script to create executable from the membrane simulator
Run this script to create a standalone .exe file
"""

import PyInstaller.__main__
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# PyInstaller arguments
PyInstaller.__main__.run([
    'interactive_simulator_compact.py',  # Main script
    '--onefile',                          # Create a single exe file
    '--windowed',                         # No console window (GUI only)
    '--name=MembraneSimulator',          # Name of the exe
    '--icon=NONE',                        # Add icon path if you have one
    '--add-data=membrane_separation.py:.',
    '--add-data=simulation_core.py:.',
    '--add-data=opex_calculator.py:.',
    '--add-data=auto_optimizer.py:.',
    '--clean',                            # Clean PyInstaller cache
])

print("\n✅ Build complete! Check the 'dist' folder for MembraneSimulator.exe")

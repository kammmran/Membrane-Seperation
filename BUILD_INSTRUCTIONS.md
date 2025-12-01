# Build Executable Instructions

## Quick Method (Recommended)

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build the EXE
```bash
python build_exe.py
```

### Step 3: Find Your EXE
The executable will be in the `dist` folder: `dist/MembraneSimulator.exe`

---

## Manual Method

If you prefer to build manually:

```bash
pyinstaller --onefile --windowed --name=MembraneSimulator interactive_simulator_compact.py
```

---

## Build Options Explained

- `--onefile` - Creates a single .exe file (easier to distribute)
- `--windowed` - No console window appears (clean GUI)
- `--name=MembraneSimulator` - Name of the output file
- `--clean` - Clean build (recommended for fresh builds)

---

## Alternative: Create Folder Distribution

For faster startup and smaller size:

```bash
pyinstaller --windowed --name=MembraneSimulator interactive_simulator_compact.py
```

This creates a folder with the .exe and supporting files.

---

## Troubleshooting

**Issue: Missing modules**
- Add them with: `--hidden-import=module_name`

**Issue: Large file size**
- Normal for Python apps (50-100MB with dependencies)
- Use folder distribution instead of `--onefile`

**Issue: Antivirus blocks the exe**
- This is normal - add exception or sign the executable

---

## Testing the EXE

1. Go to `dist` folder
2. Double-click `MembraneSimulator.exe`
3. The GUI should open without needing Python installed

---

## Distribution

You can now share the `MembraneSimulator.exe` file with anyone. They don't need Python installed!

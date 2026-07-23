# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = []
binaries += collect_dynamic_libs('PyQt6')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=[('app', 'app'), ('resources', 'resources')],
    hiddenimports=['skimage.segmentation._watershed', 'skimage.feature.peak', 'read_roi', 'multiprocessing', 'concurrent.futures', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.qt_compat'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'ultralytics', 'onnxruntime'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DropletAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources\\icon.ico'],
)

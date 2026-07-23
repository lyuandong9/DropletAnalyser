import sys, os

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        # PyInstaller onedir: _internal has everything
        if hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        else:
            base = os.path.join(os.path.dirname(sys.executable), '_internal')
        # Set Qt platform plugin path
        plugins = os.path.join(base, 'PyQt6', 'Qt6', 'plugins')
        if os.path.exists(plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins
        os.environ['QT_DEBUG_PLUGINS'] = '0'

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app.main_app import main
    main()

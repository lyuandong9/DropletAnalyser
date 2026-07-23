"""
Main application window with tabbed interface.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QMessageBox, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QProgressBar,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon

from app.gui.project_panel import ProjectPanel
from app.gui.review_panel import ReviewPanel
from app.gui.stats_panel import StatsPanel
from app.data.database import ProjectDatabase
from app.utils.config import AppConfig, get_default_config_path
from app.model.train import DropletDetector


class MainWindow(QMainWindow):
    """Main window with project, review, and statistics tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageIdentification — Droplet/Bubble Analysis")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 700)

        self.config: Optional[AppConfig] = None
        self.db: Optional[ProjectDatabase] = None
        self.detector: Optional[DropletDetector] = None
        self.project_dir: Optional[str] = None

        self._load_config()
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()

    def _load_config(self) -> None:
        config_path = get_default_config_path()
        if os.path.exists(config_path):
            self.config = AppConfig.load(config_path)
        else:
            self.config = AppConfig()

    def _save_config(self) -> None:
        if self.config:
            self.config.save(get_default_config_path())

    def _setup_ui(self) -> None:
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.project_panel = ProjectPanel(self)
        self.review_panel = ReviewPanel(self)
        self.stats_panel = StatsPanel(self)

        self.tab_widget.addTab(self.project_panel, "  项目管理 & 批量处理  ")
        self.tab_widget.addTab(self.review_panel, "  人工复核  ")
        self.tab_widget.addTab(self.stats_panel, "  统计分析 & 导出  ")

        # Initially disable review and stats tabs
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("文件(&F)")
        new_project_action = QAction("新建项目...", self)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("打开项目...", self)
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        load_model_action = QAction("加载模型权重...", self)
        load_model_action.triggered.connect(self._load_model)
        file_menu.addAction(load_model_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于...", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("就绪")
        self.statusbar.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_progress(self, value: int, maximum: int = 100) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        if value >= maximum:
            self.progress_bar.setVisible(False)

    # --- Actions ---

    def _new_project(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择项目目录", os.path.expanduser("~")
        )
        if not dir_path:
            return

        project_name = Path(dir_path).name
        self.project_dir = dir_path
        db_path = os.path.join(dir_path, "project.db")
        self.db = ProjectDatabase(db_path)
        self.db.create_project(project_name)

        self.config.project_dir = dir_path
        self._save_config()

        self.project_panel.on_project_opened()
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(2, True)
        self.tab_widget.setCurrentIndex(0)
        self.set_status(f"项目已创建: {dir_path}")

    def _open_project(self) -> None:
        db_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目数据库", os.path.expanduser("~"),
            "SQLite Database (*.db);;All Files (*)"
        )
        if not db_path:
            return

        self.project_dir = str(Path(db_path).parent)
        self.db = ProjectDatabase(db_path)
        project = self.db.get_project()
        if not project:
            QMessageBox.warning(self, "错误", "无效的项目数据库文件。")
            self.db = None
            return

        self.config.project_dir = self.project_dir
        self._save_config()

        self.project_panel.on_project_opened()
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setTabEnabled(2, True)
        self.tab_widget.setCurrentIndex(0)
        self.set_status(f"项目已打开: {self.project_dir}")

    def _load_model(self) -> None:
        weights_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重文件",
            os.path.join(os.path.dirname(__file__), "..", "model", "weights"),
            "Model Weights (*.pt *.onnx);;All Files (*)"
        )
        if not weights_path:
            return

        try:
            self.detector = DropletDetector(
                weights_path,
                conf_threshold=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold,
            )
            self.config.model_weights = weights_path
            self._save_config()
            self.set_status(f"模型已加载: {os.path.basename(weights_path)}")
            self.project_panel.on_model_loaded()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载模型:\n{e}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "关于 ImageIdentification",
            "ImageIdentification v1.0\n\n"
            "复杂多相混合图像识别系统\n"
            "用于气泡/液滴检测、粒径分析及统计\n\n"
            "Powered by YOLOv8-seg + PyQt6"
        )

    def closeEvent(self, event) -> None:
        if self.db:
            self.db.close()
        self._save_config()
        event.accept()

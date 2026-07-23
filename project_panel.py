"""
Project management and batch processing panel.
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QPushButton,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QMessageBox, QProgressBar, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

from app.data.database import ProjectDatabase
from app.engine.batch_worker import BatchWorker


class ImageThumbnailList(QListWidget):
    """List widget displaying image thumbnails with drag-drop support."""

    images_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(128, 96))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setDragEnabled(False)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setSpacing(4)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isfile(p):
                paths.append(p)
        if paths:
            self.images_dropped.emit(paths)

    def add_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                128, 96, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem()
            item.setIcon(scaled)
            item.setText(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setSizeHint(QSize(140, 120))
            self.addItem(item)

    def clear_images(self) -> None:
        self.clear()

    def selected_paths(self) -> list[str]:
        return [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).isSelected()
        ]

    def all_paths(self) -> list[str]:
        return [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
        ]


class ProjectPanel(QWidget):
    """Project management, image import, and batch processing controls."""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.worker: Optional[BatchWorker] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left panel: image list
        left = QVBoxLayout()

        # Toolbar
        btn_layout = QHBoxLayout()
        self.btn_import = QPushButton("导入图片...")
        self.btn_import.clicked.connect(self._import_images)
        self.btn_import.setEnabled(False)
        btn_layout.addWidget(self.btn_import)

        self.btn_clear = QPushButton("清空列表")
        self.btn_clear.clicked.connect(self._clear_images)
        self.btn_clear.setEnabled(False)
        btn_layout.addWidget(self.btn_clear)
        left.addLayout(btn_layout)

        self.image_list = ImageThumbnailList()
        self.image_list.images_dropped.connect(self._on_images_dropped)
        left.addWidget(self.image_list)

        self.image_count_label = QLabel("共 0 张图片")
        left.addWidget(self.image_count_label)

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Right panel: controls
        right = QVBoxLayout()

        # Scale calibration
        calib_group = QGroupBox("比例尺校准 (μm/px)")
        calib_layout = QVBoxLayout()
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 100000.0)
        self.scale_spin.setDecimals(6)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        calib_layout.addWidget(self.scale_spin)
        calib_layout.addWidget(
            QLabel("输入 μm/px 值，或在复核面板中通过划线校准")
        )
        calib_group.setLayout(calib_layout)
        right.addWidget(calib_group)

        # Detection parameters
        det_group = QGroupBox("检测参数")
        det_layout = QVBoxLayout()

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        conf_layout.addWidget(self.conf_spin)
        det_layout.addLayout(conf_layout)

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU 阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        iou_layout.addWidget(self.iou_spin)
        det_layout.addLayout(iou_layout)

        ws_layout = QHBoxLayout()
        ws_layout.addWidget(QLabel("分水岭最小距离:"))
        self.ws_spin = QSpinBox()
        self.ws_spin.setRange(5, 200)
        self.ws_spin.setValue(20)
        ws_layout.addWidget(self.ws_spin)
        det_layout.addLayout(ws_layout)

        det_group.setLayout(det_layout)
        right.addWidget(det_group)

        # Filtering
        filter_group = QGroupBox("粒径过滤")
        filter_layout = QVBoxLayout()

        min_d_layout = QHBoxLayout()
        min_d_layout.addWidget(QLabel("最小直径 (μm):"))
        self.min_d_spin = QDoubleSpinBox()
        self.min_d_spin.setRange(0.0, 100000.0)
        self.min_d_spin.setDecimals(2)
        self.min_d_spin.setValue(0.0)
        min_d_layout.addWidget(self.min_d_spin)
        filter_layout.addLayout(min_d_layout)

        max_d_layout = QHBoxLayout()
        max_d_layout.addWidget(QLabel("最大直径 (μm):"))
        self.max_d_spin = QDoubleSpinBox()
        self.max_d_spin.setRange(0.0, 100000.0)
        self.max_d_spin.setDecimals(2)
        self.max_d_spin.setValue(100000.0)
        max_d_layout.addWidget(self.max_d_spin)
        filter_layout.addLayout(max_d_layout)

        filter_group.setLayout(filter_layout)
        right.addWidget(filter_group)

        # Preprocessing
        preproc_group = QGroupBox("预处理")
        preproc_layout = QVBoxLayout()
        self.adaptive_check = QCheckBox("自适应预处理")
        self.adaptive_check.setChecked(True)
        preproc_layout.addWidget(self.adaptive_check)
        self.denoise_check = QCheckBox("去噪 (非局部均值)")
        self.denoise_check.setChecked(True)
        preproc_layout.addWidget(self.denoise_check)
        self.illum_check = QCheckBox("光照校正 (顶帽变换)")
        self.illum_check.setChecked(True)
        preproc_layout.addWidget(self.illum_check)
        preproc_group.setLayout(preproc_layout)
        right.addWidget(preproc_group)

        # Batch controls
        batch_group = QGroupBox("批量处理")
        batch_layout = QVBoxLayout()

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("处理范围:"))
        self.btn_all = QPushButton("全部")
        self.btn_all.clicked.connect(lambda: self._select_range("all"))
        range_layout.addWidget(self.btn_all)
        self.btn_selected = QPushButton("选中")
        self.btn_selected.clicked.connect(lambda: self._select_range("selected"))
        range_layout.addWidget(self.btn_selected)
        batch_layout.addLayout(range_layout)

        self.range_label = QLabel("处理: 全部图片")
        batch_layout.addWidget(self.range_label)

        self.batch_progress = QProgressBar()
        batch_layout.addWidget(self.batch_progress)
        self.batch_status = QLabel("就绪")
        batch_layout.addWidget(self.batch_status)

        btn_batch_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_start.clicked.connect(self._start_batch)
        self.btn_start.setEnabled(False)
        btn_batch_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop_batch)
        self.btn_stop.setEnabled(False)
        btn_batch_layout.addWidget(self.btn_stop)
        batch_layout.addLayout(btn_batch_layout)

        batch_group.setLayout(batch_layout)
        right.addWidget(batch_group)

        right.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(320)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._process_range = "all"

    # --- Slots ---

    def on_project_opened(self) -> None:
        self.btn_import.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self._refresh_image_list()

    def on_model_loaded(self) -> None:
        self.btn_start.setEnabled(True)

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片",
            os.path.expanduser("~"),
            "Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if files:
            self._add_images(files)

    def _on_images_dropped(self, paths: list[str]) -> None:
        self._add_images(paths)

    def _add_images(self, paths: list[str]) -> None:
        if not self.mw.db:
            return
        self.mw.db.add_images(paths)
        for p in paths:
            self.image_list.add_image(p)
        self.image_count_label.setText(f"共 {self.image_list.count()} 张图片")

    def _clear_images(self) -> None:
        self.image_list.clear_images()
        self.image_count_label.setText("共 0 张图片")

    def _refresh_image_list(self) -> None:
        self.image_list.clear_images()
        if not self.mw.db:
            return
        images = self.mw.db.get_all_images()
        for img in images:
            self.image_list.add_image(img["path"])
        self.image_count_label.setText(f"共 {self.image_list.count()} 张图片")

    def _select_range(self, range_type: str) -> None:
        self._process_range = range_type
        if range_type == "all":
            self.range_label.setText(f"处理: 全部图片 ({self.image_list.count()} 张)")
        else:
            n = len(self.image_list.selected_paths())
            self.range_label.setText(f"处理: 选中图片 ({n} 张)")

    def _start_batch(self) -> None:
        if not self.mw.detector or not self.mw.db:
            QMessageBox.warning(self, "提示", "请先加载模型并打开项目。")
            return

        if self._process_range == "all":
            paths = self.image_list.all_paths()
        else:
            paths = self.image_list.selected_paths()
            if not paths:
                QMessageBox.warning(self, "提示", "请先选择要处理的图片。")
                return

        if not paths:
            QMessageBox.warning(self, "提示", "没有可处理的图片。")
            return

        # Update detector settings
        self.mw.detector.conf_threshold = self.conf_spin.value()
        self.mw.detector.iou_threshold = self.iou_spin.value()

        preproc_params = {
            "denoise": self.denoise_check.isChecked(),
            "correct_illumination": self.illum_check.isChecked(),
            "enhance_contrast": True,
        } if not self.adaptive_check.isChecked() else {}

        postproc_params = {
            "min_distance": self.ws_spin.value(),
            "threshold_rel": 0.3,
        }

        self.worker = BatchWorker(
            image_paths=paths,
            detector=self.mw.detector,
            um_per_pixel=self.scale_spin.value(),
            adaptive_preprocess=self.adaptive_check.isChecked(),
            preprocess_params=preproc_params,
            postprocess_params=postproc_params,
            min_diameter_um=self.min_d_spin.value(),
            max_diameter_um=self.max_d_spin.value(),
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.progress_detail.connect(self._on_detail)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.batch_finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.batch_progress.setMaximum(len(paths))
        self.batch_progress.setValue(0)

        self.worker.start()

    def _stop_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.batch_status.setText("正在停止...")

    def _on_progress(self, idx: int) -> None:
        self.batch_progress.setValue(idx + 1)

    def _on_detail(self, msg: str) -> None:
        self.batch_status.setText(msg)
        self.mw.set_status(msg)

    def _on_image_done(self, idx: int, features: list[dict]) -> None:
        if not self.mw.db:
            return
        img_path = self.image_list.all_paths()[idx] if idx < self.image_list.count() else None
        if img_path:
            img_record = self.mw.db.get_image_by_path(img_path)
            if img_record:
                self.mw.db.save_droplets(img_record["id"], features)
                self.mw.db.update_image_status(
                    img_record["id"], "processed", len(features)
                )

    def _on_finished(self, stats: dict) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.batch_progress.setValue(0)

        cancelled = stats.get("_cancelled", False)
        msg = (
            f"处理{'已取消' if cancelled else '完成'}！\n"
            f"处理图片: {stats['processed_images']} / {stats['total_images']}\n"
            f"检测到总数: {stats['total_droplets']} 个\n"
            f"错误: {stats['errors']} 张"
        )
        self.batch_status.setText(msg)
        self.mw.set_status(msg.replace("\n", " | "))

        # Refresh review panel
        self.mw.review_panel.on_batch_finished()
        self.mw.stats_panel.refresh()

        if not cancelled:
            QMessageBox.information(self, "处理完成", msg)

    def _on_error(self, idx: int, error: str) -> None:
        print(f"Error processing image {idx}: {error}")

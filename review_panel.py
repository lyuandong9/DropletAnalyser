"""
Manual review and correction panel for detection results.
"""

import json
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QMessageBox, QSlider, QFrame, QHeaderView,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QMouseEvent, QWheelEvent, QKeyEvent, QCursor,
)

from app.engine.features import equivalent_circle_diameter, centroid, extract_droplet_features

# Detection status colors
STATUS_COLORS = {
    "pending": QColor(200, 200, 200),
    "processed": QColor(255, 200, 100),
    "reviewed": QColor(100, 200, 100),
}

# Contour colors
AUTO_COLOR = QColor(0, 180, 255, 180)  # Blue for auto-detected
MANUAL_COLOR = QColor(255, 180, 0, 180)  # Orange for manually added
SELECTED_COLOR = QColor(255, 50, 50, 220)  # Red for selected
ID_FONT = QFont("Arial", 8, QFont.Weight.Bold)


class ImageViewer(QWidget):
    """Scrollable, zoomable image viewer with contour overlay and interaction."""

    droplet_selected = pyqtSignal(int)  # local_id
    droplet_deleted = pyqtSignal(int)
    droplet_added = pyqtSignal(dict)  # contour points
    contour_updated = pyqtSignal(int, list)  # local_id, new contour

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.droplets: list[dict] = []
        self.scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.selected_id: Optional[int] = None

        # Drawing mode
        self.drawing: bool = False
        self.drawing_points: list[tuple[int, int]] = []
        self.edit_mode: str = "none"  # none, draw, adjust

        # Pan state
        self.panning: bool = False
        self.pan_start: tuple[int, int] = (0, 0)

    def load_image(self, image_path: str) -> None:
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selected_id = None
        self.drawing_points = []
        self.edit_mode = "none"
        self._update_display()
        self.update()

    def set_droplets(self, droplets: list[dict]) -> None:
        self.droplets = droplets
        self.update()

    def set_scale(self, scale: float) -> None:
        self.scale = max(0.05, min(10.0, scale))
        self.update()

    def _update_display(self) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.display_image = resized

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.display_image is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw image
        h, w, c = self.display_image.shape
        qimg = QImage(
            self.display_image.data, w, h, w * 3, QImage.Format.Format_RGB888
        )
        painter.drawImage(self.offset_x, self.offset_y, qimg)

        # Draw droplets
        for d in self.droplets:
            self._draw_droplet(painter, d)

        # Draw in-progress polygon
        if self.edit_mode == "draw" and len(self.drawing_points) >= 2:
            pen = QPen(QColor(0, 255, 0, 200), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            for i in range(len(self.drawing_points) - 1):
                p1 = self._img_to_widget(self.drawing_points[i])
                p2 = self._img_to_widget(self.drawing_points[i + 1])
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

    def _draw_droplet(self, painter: QPainter, d: dict) -> None:
        is_selected = d.get("local_id") == self.selected_id
        is_manual = d.get("is_manual", False)

        if is_selected:
            color = SELECTED_COLOR
            width = 3
        elif is_manual:
            color = MANUAL_COLOR
            width = 2
        else:
            color = AUTO_COLOR
            width = 1

        pen = QPen(color, width)
        painter.setPen(pen)

        # Draw contour
        contour = d.get("contour_raw")
        if contour and len(contour) >= 3:
            pts = [self._img_to_widget(tuple(p)) for p in contour]
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        elif "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            p1 = self._img_to_widget((x1, y1))
            p2 = self._img_to_widget((x2, y2))
            painter.drawRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])

        # Draw ID label
        cx = d.get("centroid_x", 0)
        cy = d.get("centroid_y", 0)
        pos = self._img_to_widget((cx, cy))
        painter.setFont(ID_FONT)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            pos[0] - 10, pos[1] + 4, str(d.get("local_id", ""))
        )

    def _img_to_widget(self, pt: tuple) -> tuple[int, int]:
        """Convert image coordinates to widget coordinates."""
        x, y = pt
        return (int(x * self.scale) + self.offset_x,
                int(y * self.scale) + self.offset_y)

    def _widget_to_img(self, pt: tuple) -> tuple[int, int]:
        """Convert widget coordinates to image coordinates."""
        x, y = pt
        return (int((x - self.offset_x) / self.scale),
                int((y - self.offset_y) / self.scale))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.image is None:
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = (event.pos().x(), event.pos().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        img_pt = self._widget_to_img((event.pos().x(), event.pos().y()))

        if event.button() == Qt.MouseButton.LeftButton:
            if self.edit_mode == "draw":
                self.drawing_points.append(img_pt)
                self.update()
            elif self.edit_mode == "adjust":
                self._select_droplet_at(img_pt)
            else:
                self._select_droplet_at(img_pt)

        elif event.button() == Qt.MouseButton.RightButton:
            if self.edit_mode == "draw" and len(self.drawing_points) >= 3:
                self._finish_drawing()
            elif self.edit_mode == "adjust" and self.selected_id is not None:
                pass  # Context menu

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.panning:
            dx = event.pos().x() - self.pan_start[0]
            dy = event.pos().y() - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.pos().x(), event.pos().y())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = self.scale * factor

        # Zoom towards cursor
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        img_x = (mouse_x - self.offset_x) / self.scale
        img_y = (mouse_y - self.offset_y) / self.scale

        self.scale = max(0.05, min(10.0, new_scale))
        self.offset_x = int(mouse_x - img_x * self.scale)
        self.offset_y = int(mouse_y - img_y * self.scale)
        self._update_display()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete and self.selected_id is not None:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Escape:
            self.edit_mode = "none"
            self.drawing_points = []
            self.update()
        elif event.key() == Qt.Key.Key_D:
            self.edit_mode = "draw"
            self.drawing_points = []
        elif event.key() == Qt.Key.Key_A:
            self.edit_mode = "adjust"

    def _select_droplet_at(self, img_pt: tuple[int, int]) -> None:
        for d in self.droplets:
            contour = d.get("contour_raw")
            if contour is not None:
                pts = np.array(contour, dtype=np.int32)
                if cv2.pointPolygonTest(pts, img_pt, False) >= 0:
                    self.selected_id = d["local_id"]
                    self.droplet_selected.emit(d["local_id"])
                    self.update()
                    return
        self.selected_id = None
        self.droplet_selected.emit(-1)
        self.update()

    def _finish_drawing(self) -> None:
        if len(self.drawing_points) < 3:
            return
        self.droplet_added.emit({
            "points": self.drawing_points,
        })
        self.drawing_points = []
        self.edit_mode = "none"
        self.update()

    def delete_selected(self) -> None:
        if self.selected_id is not None:
            self.droplet_deleted.emit(self.selected_id)
            self.selected_id = None
            self.update()

    def fit_to_window(self) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        win_w = self.width() - 20
        win_h = self.height() - 20
        self.scale = min(win_w / w, win_h / h, 1.0)
        self.offset_x = (win_w - int(w * self.scale)) // 2
        self.offset_y = (win_h - int(h * self.scale)) // 2
        self._update_display()
        self.update()


class ReviewPanel(QWidget):
    """Manual review panel with image viewer and droplet table."""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.current_image_idx: int = 0
        self.image_paths: list[str] = []
        self.current_droplets: list[dict] = []
        self.current_masks: dict[int, np.ndarray] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Center: Image viewer
        center = QVBoxLayout()

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self._prev_image)
        toolbar.addWidget(self.btn_prev)

        self.image_label = QLabel("未加载图片")
        toolbar.addWidget(self.image_label, 1)

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self._next_image)
        toolbar.addWidget(self.btn_next)

        toolbar.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["浏览", "绘制(D)", "调整(A)"])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        toolbar.addWidget(self.mode_combo)

        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self._fit)
        toolbar.addWidget(self.btn_fit)

        self.btn_delete = QPushButton("删除选中 (Del)")
        self.btn_delete.clicked.connect(self._delete_selected)
        toolbar.addWidget(self.btn_delete)

        self.status_label = QLabel("")
        toolbar.addWidget(self.status_label)
        center.addLayout(toolbar)

        self.viewer = ImageViewer()
        self.viewer.droplet_selected.connect(self._on_droplet_selected)
        self.viewer.droplet_deleted.connect(self._on_droplet_deleted)
        self.viewer.droplet_added.connect(self._on_droplet_added)
        center.addWidget(self.viewer, 1)

        center_widget = QWidget()
        center_widget.setLayout(center)

        # Right: Droplet table
        right = QVBoxLayout()

        table_label = QLabel("检测结果")
        table_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        right.addWidget(table_label)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID", "直径(μm)", "面积(px)", "圆度", "置信度", "来源"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._table_selection_changed)
        right.addWidget(self.table)

        # Info
        info_group = QFrame()
        info_group.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout()
        self.info_label = QLabel("选择液滴查看详情")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        right.addWidget(info_group)

        # Mark reviewed
        self.btn_reviewed = QPushButton("✓ 标记为已审核")
        self.btn_reviewed.clicked.connect(self._mark_reviewed)
        right.addWidget(self.btn_reviewed)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(380)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def on_batch_finished(self) -> None:
        """Populate image list from database after batch processing."""
        if not self.mw.db:
            return
        images = self.mw.db.get_all_images()
        self.image_paths = [img["path"] for img in images if img["status"] != "pending"]
        if self.image_paths:
            self.current_image_idx = 0
            self._load_current_image()
        else:
            self.image_label.setText("无已处理图片")

    def _load_current_image(self) -> None:
        if not self.image_paths:
            return

        path = self.image_paths[self.current_image_idx]
        self.image_label.setText(
            f" [{self.current_image_idx + 1} / {len(self.image_paths)}]  "
            f"{path}"
        )
        self.viewer.load_image(path)
        self.viewer.fit_to_window()

        # Load droplets from DB
        if self.mw.db:
            img_record = self.mw.db.get_image_by_path(path)
            if img_record:
                droplets = self.mw.db.get_droplets_for_image(img_record["id"])
                self.current_droplets = droplets

                # Parse contour data
                for d in self.current_droplets:
                    contour_json = d.get("contour_json", "[]")
                    try:
                        d["contour_raw"] = json.loads(contour_json)
                    except (json.JSONDecodeError, TypeError):
                        d["contour_raw"] = []

                    d["is_manual"] = d.get("is_manual", 0) == 1
                    d["local_id"] = d["local_id"]

                self.viewer.set_droplets(self.current_droplets)
                self._populate_table()

                status = img_record.get("status", "pending")
                self.status_label.setText(f"状态: {status}")
            else:
                self.current_droplets = []
                self.viewer.set_droplets([])
                self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.current_droplets))
        for i, d in enumerate(self.current_droplets):
            self.table.setItem(i, 0, QTableWidgetItem(str(d.get("local_id", i + 1))))
            self.table.setItem(i, 1, QTableWidgetItem(f"{d.get('eq_diameter_um', 0):.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(str(d.get("area_px", 0))))
            self.table.setItem(i, 3, QTableWidgetItem(f"{d.get('circularity', 0):.3f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{d.get('confidence', 0):.3f}"))
            source = "手动" if d.get("is_manual", False) else "自动"
            self.table.setItem(i, 5, QTableWidgetItem(source))

    def _prev_image(self) -> None:
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_current_image()

    def _next_image(self) -> None:
        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self._load_current_image()

    def _mode_changed(self, idx: int) -> None:
        modes = ["none", "draw", "adjust"]
        self.viewer.edit_mode = modes[idx]

    def _fit(self) -> None:
        self.viewer.fit_to_window()

    def _delete_selected(self) -> None:
        self.viewer.delete_selected()

    def _on_droplet_selected(self, local_id: int) -> None:
        # Select in table
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).text() == str(local_id):
                self.table.selectRow(i)
                break
        # Update info
        for d in self.current_droplets:
            if d.get("local_id") == local_id:
                self._show_droplet_info(d)
                return
        self.info_label.setText("选择液滴查看详情")

    def _on_droplet_deleted(self, local_id: int) -> None:
        if not self.mw.db:
            return
        # Find and delete from DB
        for d in self.current_droplets:
            if d.get("local_id") == local_id:
                self.mw.db.delete_droplet(d["id"])
                break
        self._load_current_image()

    def _on_droplet_added(self, data: dict) -> None:
        if not self.mw.db or not self.viewer.image is not None:
            return
        path = self.image_paths[self.current_image_idx]
        img_record = self.mw.db.get_image_by_path(path)
        if not img_record:
            return

        points = data["points"]
        # Create mask from polygon
        h, w = self.viewer.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        area = int(np.sum(mask))
        bbox = cv2.boundingRect(pts) if contour is not None else [0, 0, 0, 0]
        cy, cx = centroid(mask)
        diam_px = equivalent_circle_diameter(area)

        um_per_pixel = self.mw.config.um_per_pixel if self.mw.config else 1.0

        # Generate new local ID
        existing_ids = {d.get("local_id", 0) for d in self.current_droplets}
        new_local_id = max(existing_ids, default=0) + 1

        contour_list = contour.squeeze(1).tolist() if contour is not None else []

        new_droplet_id = self.mw.db.insert_manual_droplet(
            image_id=img_record["id"],
            local_id=new_local_id,
            area_px=area,
            eq_diameter_px=round(diam_px, 4),
            eq_diameter_um=round(diam_px * um_per_pixel, 4),
            centroid_x=round(cx, 4),
            centroid_y=round(cy, 4),
            bbox_x1=int(bbox[0]),
            bbox_y1=int(bbox[1]),
            bbox_x2=int(bbox[0] + bbox[2]),
            bbox_y2=int(bbox[1] + bbox[3]),
            contour_json=json.dumps(contour_list),
        )

        self._load_current_image()

    def _table_selection_changed(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.viewer.selected_id = None
            self.viewer.update()
            return
        row = selected[0].row()
        local_id = int(self.table.item(row, 0).text())
        self.viewer.selected_id = local_id
        self.viewer.update()
        for d in self.current_droplets:
            if d.get("local_id") == local_id:
                self._show_droplet_info(d)
                return

    def _show_droplet_info(self, d: dict) -> None:
        info = (
            f"ID: {d.get('local_id', '')}\n"
            f"等效直径: {d.get('eq_diameter_um', 0):.2f} μm\n"
            f"面积: {d.get('area_px', 0)} px\n"
            f"圆度: {d.get('circularity', 0):.4f}\n"
            f"离心率: {d.get('eccentricity', 0):.4f}\n"
            f"长轴: {d.get('major_axis_um', 0):.2f} μm\n"
            f"短轴: {d.get('minor_axis_um', 0):.2f} μm\n"
            f"置信度: {d.get('confidence', 0):.3f}\n"
            f"来源: {'手动' if d.get('is_manual', False) else '自动'}"
        )
        self.info_label.setText(info)

    def _mark_reviewed(self) -> None:
        if not self.mw.db or not self.image_paths:
            return
        path = self.image_paths[self.current_image_idx]
        img_record = self.mw.db.get_image_by_path(path)
        if img_record:
            self.mw.db.update_image_status(img_record["id"], "reviewed")
            self.status_label.setText("状态: reviewed")

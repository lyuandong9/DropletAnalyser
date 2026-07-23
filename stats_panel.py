"""
Statistical analysis panel with charts and export controls.
"""

import os
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding plots in PyQt6."""

    def __init__(self):
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        super().__init__(self.fig)
        self.fig.tight_layout(pad=3.0)


class StatsPanel(QWidget):
    """Statistics display and export panel."""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Left: Summary stats + Export
        left = QVBoxLayout()

        # Summary group
        summary_group = QGroupBox("汇总统计")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table.setMaximumHeight(250)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)
        left.addWidget(summary_group)

        # Distribution controls
        dist_group = QGroupBox("粒径分布设置")
        dist_layout = QVBoxLayout()
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bin 数量:"))
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(5, 200)
        self.bin_spin.setValue(50)
        self.bin_spin.valueChanged.connect(self.refresh)
        bin_layout.addWidget(self.bin_spin)
        dist_layout.addLayout(bin_layout)
        dist_group.setLayout(dist_layout)
        left.addWidget(dist_group)

        # Export
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout()
        self.btn_export_excel = QPushButton("导出 Excel 综合报告...")
        self.btn_export_excel.clicked.connect(self._export_excel)
        export_layout.addWidget(self.btn_export_excel)
        self.btn_export_csv = QPushButton("导出 CSV 数据...")
        self.btn_export_csv.clicked.connect(self._export_csv)
        export_layout.addWidget(self.btn_export_csv)
        self.btn_export_images = QPushButton("导出标注图片...")
        self.btn_export_images.clicked.connect(self._export_images)
        export_layout.addWidget(self.btn_export_images)
        export_group.setLayout(export_layout)
        left.addWidget(export_group)

        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)

        # Right: Charts
        right = QVBoxLayout()
        self.canvas = MplCanvas()
        right.addWidget(self.canvas)

        right_widget = QWidget()
        right_widget.setLayout(right)

        layout.addWidget(left_widget)
        layout.addWidget(right_widget, 1)

    def refresh(self) -> None:
        if not self.mw.db:
            return
        self._update_summary()
        self._update_charts()

    def _update_summary(self) -> None:
        stats = self.mw.db.get_statistics()
        items = [
            ("总检测数", str(stats["total_count"])),
            ("平均直径 (μm)", f"{stats['mean_diameter_um']:.2f}"),
            ("中位直径 (μm)", f"{stats['median_diameter_um']:.2f}"),
            ("Sauter 平均直径 D32 (μm)", f"{stats['d32_um']:.2f}"),
            ("手动修正数", str(stats["manual_count"])),
        ]
        self.summary_table.setRowCount(len(items))
        for i, (label, value) in enumerate(items):
            self.summary_table.setItem(i, 0, QTableWidgetItem(label))
            self.summary_table.setItem(i, 1, QTableWidgetItem(value))

    def _update_charts(self) -> None:
        bins = self.bin_spin.value()
        distribution = self.mw.db.get_diameter_distribution(bins)

        if not distribution:
            self.canvas.ax1.clear()
            self.canvas.ax2.clear()
            self.canvas.draw()
            return

        bin_centers = [(d["bin_start"] + d["bin_end"]) / 2 for d in distribution]
        counts = [d["count"] for d in distribution]

        total = sum(counts)
        cumulative = np.cumsum(counts) / total * 100 if total > 0 else []

        # Histogram
        self.canvas.ax1.clear()
        self.canvas.ax1.bar(
            bin_centers, counts,
            width=(bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1,
            color="steelblue", edgecolor="white", alpha=0.8,
        )
        self.canvas.ax1.set_xlabel("Equivalent Diameter (μm)")
        self.canvas.ax1.set_ylabel("Count")
        self.canvas.ax1.set_title("Particle Size Distribution")

        # Cumulative
        self.canvas.ax2.clear()
        self.canvas.ax2.plot(bin_centers, cumulative, "r-", linewidth=2)
        self.canvas.ax2.set_xlabel("Equivalent Diameter (μm)")
        self.canvas.ax2.set_ylabel("Cumulative (%)")
        self.canvas.ax2.set_title("Cumulative Distribution")
        self.canvas.ax2.set_ylim(0, 105)
        self.canvas.ax2.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    # --- Export ---

    def _export_excel(self) -> None:
        if not self.mw.db:
            QMessageBox.warning(self, "提示", "请先打开项目。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel 报告",
            os.path.join(self.mw.project_dir or os.path.expanduser("~"), "report.xlsx"),
            "Excel Files (*.xlsx)"
        )
        if not path:
            return

        try:
            from app.data.export_excel import export_excel_report
            export_excel_report(self.mw.db, path)
            QMessageBox.information(self, "导出完成", f"报告已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_csv(self) -> None:
        if not self.mw.db:
            QMessageBox.warning(self, "提示", "请先打开项目。")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV 数据",
            os.path.join(self.mw.project_dir or os.path.expanduser("~"), "droplets.csv"),
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            from app.data.export_excel import export_csv_data
            export_csv_data(self.mw.db, path)
            QMessageBox.information(self, "导出完成", f"数据已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_images(self) -> None:
        if not self.mw.db:
            QMessageBox.warning(self, "提示", "请先打开项目。")
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "选择导出目录",
            self.mw.project_dir or os.path.expanduser("~")
        )
        if not dir_path:
            return

        try:
            from app.data.export_excel import export_annotated_images
            count = export_annotated_images(self.mw.db, dir_path)
            QMessageBox.information(
                self, "导出完成",
                f"共导出 {count} 张标注图片到:\n{dir_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

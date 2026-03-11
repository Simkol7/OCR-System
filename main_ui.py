import sys
import cv2
import os
import numpy as np  # 导入 numpy 辅助图像检查
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QRadioButton, QTextEdit,
    QGroupBox, QMessageBox, QSlider, QDesktopWidget, QSizePolicy,
    QFrame,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB

# =========================================================
# 🎨 动态字号 QSS 模板
# =========================================================
def build_qss(b: int) -> str:
    xs = max(b - 4, 9)  # xsmall
    s = max(b - 2, 10)  # small
    m = b  # medium
    l = b + 2  # large

    return f"""
/* ── 全局 ── */
QMainWindow, QWidget {{
    background-color: #0D1117;
    font-family: "JetBrains Mono", "Consolas", "Microsoft YaHei UI", monospace;
    color: #C9D1D9;
}}

/* ── GroupBox ── */
QGroupBox {{
    font-size: {xs}px;
    font-weight: bold;
    letter-spacing: 2px;
    border: 1px solid #21262D;
    border-top: 2px solid #1F6FEB;
    border-radius: 6px;
    margin-top: 24px;
    padding-top: 22px;
    padding-left: 14px;
    padding-right: 14px;
    padding-bottom: 14px;
    background-color: #161B22;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #58A6FF;
    background-color: #161B22;
}}

/* ── 通用 QLabel ── */
QLabel {{
    font-size: {m}px;
    color: #8B949E;
    background: transparent;
}}

/* ── 图像展示区占位 ── */
QLabel#lbl_image_display {{
    background-color: #0D1117;
    border: 1px solid #21262D;
    border-radius: 8px;
    font-size: {m}px;
    color: #30363D;
    font-weight: bold;
}}

/* ── 导航栏 ── */
QLabel#lbl_nav_title {{
    font-size: {l}px;
    font-weight: bold;
    color: #E6EDF3;
    letter-spacing: 1px;
}}
QLabel#lbl_nav_subtitle {{
    font-size: {s}px;
    color: #484F58;
    letter-spacing: 0.5px;
}}

/* ── 字号控件 ── */
QLabel#lbl_fontsize_sm {{ font-size: {s}px; color: #484F58; }}
QLabel#lbl_fontsize_lg {{ font-size: {l}px; color: #8B949E; }}
QLabel#lbl_fontsize_val {{ font-size: {s}px; color: #58A6FF; font-weight: bold; min-width: 34px; }}

/* ── 图区 标签 ── */
QLabel#lbl_img_tag {{ font-size: {xs}px; color: #484F58; letter-spacing: 3px; font-weight: bold; }}

/* ── 状态栏 ── */
QLabel#lbl_time {{ font-size: {s}px; color: #484F58; letter-spacing: 0.5px; }}
QLabel#lbl_path {{ font-size: {s}px; color: #30363D; }}

/* ── 参数滑块标签 ── */
QLabel#slider_label {{ font-size: {m}px; color: #8B949E; }}
QLabel#slider_value {{ font-size: {m}px; color: #58A6FF; font-weight: bold; }}

/* ── 单选按钮 ── */
QRadioButton {{ font-size: {m}px; color: #C9D1D9; padding: 6px 0; spacing: 10px; }}
QRadioButton::indicator {{ width: {m}px; height: {m}px; border-radius: {m // 2}px; border: 2px solid #30363D; background: #0D1117; }}
QRadioButton::indicator:checked {{ background: #1F6FEB; border-color: #58A6FF; }}

/* ── 普通按钮 ── */
QPushButton {{ border: 1px solid #30363D; border-radius: 6px; padding: 9px 16px; background-color: #21262D; color: #C9D1D9; font-size: {m}px; }}
QPushButton#btn_run_main {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F6FEB, stop:1 #0E4DAB); border: none; color: #FFFFFF; font-weight: bold; font-size: {l}px; padding: 14px; }}
QPushButton#btn_export_main {{ background: transparent; border: 1px solid #238636; color: #3FB950; font-size: {m}px; }}

/* ── 文本输出 ── */
QTextEdit {{ border: 1px solid #21262D; border-radius: 6px; background-color: #0D1117; padding: 14px; font-size: {l}px; color: #C9D1D9; font-family: "JetBrains Mono", monospace; }}

/* ── 参数滑块 ── */
QSlider::groove:horizontal {{ border: none; background: #21262D; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #FFFFFF; border: 2px solid #1F6FEB; width: 12px; height: 12px; margin-top: -5px; margin-bottom: -5px; border-radius: 6px; }}
"""


# ── 工作线程 ──────────────────────────────────────────────────────────────
class ModelLoaderWorker(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, algo_b):
        super().__init__()
        self.algo_b = algo_b

    def run(self):
        self.algo_b.load_model_if_needed()
        self.finished_signal.emit()


class OCRWorker(QThread):
    finished_signal = pyqtSignal(str, object, float, str)

    def __init__(self, algo_type, img_path, algo_a, algo_b, threshold, erode_sz, dilate_x, dilate_y, unclip_ratio):
        super().__init__()
        self.algo_type, self.img_path = algo_type, img_path
        self.algo_a, self.algo_b = algo_a, algo_b
        self.threshold, self.erode_sz = threshold, erode_sz
        self.dilate_x, self.dilate_y = dilate_x, dilate_y
        self.unclip_ratio = unclip_ratio

    def run(self):
        result_text, result_img, error_msg, cost_time = "", None, "", 0.0
        try:
            if self.algo_type == 'A':
                result_text, result_img, cost_time = self.algo_a.detect_and_recognize(
                    self.img_path, conf_threshold=self.threshold,
                    erode_size=self.erode_sz, dilate_x=self.dilate_x, dilate_y=self.dilate_y)
            else:
                result_text, result_img, cost_time = self.algo_b.detect_and_recognize(
                    self.img_path, conf_threshold=self.threshold, unclip_ratio=self.unclip_ratio)
        except Exception as e:
            error_msg = str(e)
        self.finished_signal.emit(result_text, result_img, cost_time, error_msg)


# ── 带标签的参数滑块 ─────────────────────────────────────────────────────
class LabeledSlider(QWidget):
    def __init__(self, label, min_val, max_val, default, formatter=None, parent=None):
        super().__init__(parent)
        self.formatter = formatter or (lambda v: str(v))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lbl = QLabel(label);
        self.lbl.setObjectName("slider_label")
        self.lbl.setMinimumWidth(60);
        self.lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val);
        self.slider.setValue(default)

        self.val_lbl = QLabel(self.formatter(default));
        self.val_lbl.setObjectName("slider_value")
        self.val_lbl.setMinimumWidth(36);
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider.valueChanged.connect(lambda v: self.val_lbl.setText(self.formatter(v)))
        layout.addWidget(self.lbl);
        layout.addWidget(self.slider);
        layout.addWidget(self.val_lbl)

    def value(self): return self.slider.value()


# ── 状态徽章 ─────────────────────────────────────────────────────────────
class StatusBadge(QLabel):
    def __init__(self, text, color="#484F58", parent=None):
        super().__init__(text, parent)
        self._color = color;
        self._apply()

    def set_color(self, color): self._color = color; self._apply()

    def _apply(self):
        self.setStyleSheet(
            f"border: 1px solid {self._color}; border-radius: 10px; padding: 2px 12px; color: {self._color}; font-weight: bold;")


# ── 主界面 ────────────────────────────────────────────────────────────────
class OCRSystemUI(QMainWindow):
    DEFAULT_FONT_SIZE = 20

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR · 智能文字识别系统")
        self._font_size = self.DEFAULT_FONT_SIZE
        self.resize(1480, 920);
        self.center_on_screen()

        self.algo_a = AlgorithmA();
        self.algo_b = AlgorithmB()
        self.current_image_path = None;
        self.current_result_img = None

        self.init_ui();
        self.apply_font_size(self._font_size)

        self.loader_thread = ModelLoaderWorker(self.algo_b)
        self.loader_thread.finished_signal.connect(self.on_model_loaded);
        self.loader_thread.start()

        self._badge_tick = 0;
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_loading);
        self._pulse_timer.start(600)

    def center_on_screen(self):
        qr = self.frameGeometry();
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp);
        self.move(qr.topLeft())

    def apply_font_size(self, size: int):
        self._font_size = size;
        self.setStyleSheet(build_qss(size))
        if hasattr(self, '_sidebar'): self._sidebar.setMinimumWidth(self._font_size * 18)
        if hasattr(self, 'status_badge'): self.status_badge._apply()

    def init_ui(self):
        root = QWidget();
        self.setCentralWidget(root)
        rl = QVBoxLayout(root);
        rl.setContentsMargins(0, 0, 0, 0);
        rl.setSpacing(0)
        rl.addWidget(self._build_navbar())
        div = QFrame();
        div.setStyleSheet("background:#21262D; max-height:1px;");
        rl.addWidget(div)

        body = QWidget();
        body.setStyleSheet("background:#0D1117;")
        bl = QHBoxLayout(body);
        bl.setContentsMargins(0, 0, 0, 0);
        bl.setSpacing(0)
        bl.addWidget(self._build_sidebar())
        vd = QFrame();
        vd.setStyleSheet("background:#21262D; max-width:1px;");
        bl.addWidget(vd)
        bl.addWidget(self._build_right_panel(), stretch=1)
        rl.addWidget(body, stretch=1);
        rl.addWidget(self._build_statusbar())

    def _build_navbar(self):
        bar = QWidget();
        bar.setFixedHeight(58);
        bar.setStyleSheet("background:#161B22;")
        layout = QHBoxLayout(bar);
        layout.setContentsMargins(20, 0, 20, 0);
        layout.setSpacing(0)
        title = QLabel("OCR / 智能文字识别");
        title.setObjectName("lbl_nav_title")
        subtitle = QLabel("  ·  双引擎形态学与深度学习");
        subtitle.setObjectName("lbl_nav_subtitle")
        layout.addWidget(title);
        layout.addWidget(subtitle);
        layout.addStretch()

        self.slider_fontsize = QSlider(Qt.Horizontal);
        self.slider_fontsize.setRange(20, 28)
        self.slider_fontsize.setValue(self.DEFAULT_FONT_SIZE);
        self.slider_fontsize.setFixedWidth(120)
        self.slider_fontsize.valueChanged.connect(self._on_fontsize_changed)
        self.lbl_fs_val = QLabel(f"{self.DEFAULT_FONT_SIZE}px");
        self.lbl_fs_val.setObjectName("lbl_fontsize_val")

        layout.addWidget(QLabel("A"));
        layout.addWidget(self.slider_fontsize);
        layout.addWidget(self.lbl_fs_val)
        self.status_badge = StatusBadge("⬤  MODEL LOADING", "#484F58");
        layout.addWidget(self.status_badge)
        return bar

    def _on_fontsize_changed(self, val: int):
        self.lbl_fs_val.setText(f"{val}px");
        self.apply_font_size(val)

    def _build_sidebar(self):
        sidebar = QWidget();
        self._sidebar = sidebar
        sidebar.setMinimumWidth(self._font_size * 18);
        sidebar.setStyleSheet("background:#0D1117;")
        layout = QVBoxLayout(sidebar);
        layout.setContentsMargins(16, 16, 16, 16);
        layout.setSpacing(14)

        self.btn_load = QPushButton("＋  选择本地图像");
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)

        algo_box = QGroupBox("ENGINE");
        al = QVBoxLayout()
        self.radio_a = QRadioButton("方案 A  —  OpenCV 形态学");
        self.radio_b = QRadioButton("方案 B  —  PaddleOCR 深度学习")
        self.radio_b.setChecked(True);
        self.radio_a.toggled.connect(self.toggle_params)
        al.addWidget(self.radio_a);
        al.addWidget(self.radio_b);
        algo_box.setLayout(al);
        layout.addWidget(algo_box)

        self.slider_conf = LabeledSlider("置信度", 10, 95, 50, lambda v: f"{v / 100:.2f}")
        layout.addWidget(self.slider_conf)

        self.group_a_params = QGroupBox("ALGO A  —  MORPHOLOGY");
        apl = QVBoxLayout()
        self.sl_dilate_x = LabeledSlider("横向膨胀", 1, 40, 15)
        self.sl_erode = LabeledSlider("腐蚀去噪", 1, 5, 2)

        # 🔴 修改点：绑定滑动条实时预览事件
        self.sl_dilate_x.slider.valueChanged.connect(self.update_morphology_preview)
        self.sl_erode.slider.valueChanged.connect(self.update_morphology_preview)

        apl.addWidget(self.sl_dilate_x);
        apl.addWidget(self.sl_erode)
        self.group_a_params.setLayout(apl);
        layout.addWidget(self.group_a_params)

        self.group_b_params = QGroupBox("ALGO B  —  DBNET");
        bpl = QVBoxLayout()
        self.sl_unclip = LabeledSlider("Vatti 膨胀率", 10, 30, 15, lambda v: f"{v / 10:.1f}")
        bpl.addWidget(self.sl_unclip);
        self.group_b_params.setLayout(bpl);
        layout.addWidget(self.group_b_params)

        layout.addStretch()
        self.btn_run = QPushButton("▶  RUN RECOGNITION");
        self.btn_run.setObjectName("btn_run_main")
        self.btn_run.setMinimumHeight(50);
        self.btn_run.clicked.connect(self.start_recognition)
        layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("↓  导出识别结果");
        self.btn_export.setObjectName("btn_export_main")
        self.btn_export.clicked.connect(self.export_results);
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)
        self.toggle_params();
        return sidebar

    def _build_right_panel(self):
        panel = QWidget();
        layout = QVBoxLayout(panel);
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setSpacing(0)
        img_row = QWidget();
        img_row.setFixedHeight(420);
        img_layout = QHBoxLayout(img_row)
        self.lbl_origin = self._make_image_pane("原始图像  /  ORIGINAL")
        self.lbl_result = self._make_image_pane("识别结果  /  RESULT")
        img_layout.addWidget(self._wrap_image_pane(self.lbl_origin, "INPUT"))
        img_layout.addWidget(self._wrap_image_pane(self.lbl_result, "OUTPUT"))
        layout.addWidget(img_row, stretch=5)

        log_row = QWidget();
        log_layout = QVBoxLayout(log_row)
        self.text_edit = QTextEdit();
        self.text_edit.setReadOnly(True)
        log_layout.addWidget(QLabel("RECOGNITION OUTPUT"));
        log_layout.addWidget(self.text_edit)
        layout.addWidget(log_row, stretch=2);
        return panel

    def _make_image_pane(self, text):
        lbl = QLabel(text);
        lbl.setObjectName("lbl_image_display");
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding);
        lbl.setMinimumHeight(200)
        return lbl

    def _wrap_image_pane(self, lbl, tag_text):
        wrapper = QWidget();
        wl = QVBoxLayout(wrapper);
        tag = QLabel(tag_text);
        tag.setObjectName("lbl_img_tag")
        wl.addWidget(tag);
        wl.addWidget(lbl);
        return wrapper

    def _build_statusbar(self):
        bar = QWidget();
        bar.setFixedHeight(32);
        layout = QHBoxLayout(bar)
        self.lbl_time = QLabel("⏱  —");
        self.lbl_path = QLabel("未选择图像")
        layout.addWidget(self.lbl_time);
        layout.addStretch();
        layout.addWidget(self.lbl_path);
        return bar

    # ── 逻辑 ─────────────────────────────────────────────────────────────
    # 🔴 修改点：实时预览函数逻辑
    def update_morphology_preview(self):
        if self.radio_a.isChecked() and self.current_image_path:
            # 调用 AlgorithmA 的快速预览方法（不含 OCR 识别）
            preview_img = self.algo_a.get_morphology_preview(
                self.current_image_path,
                self.sl_erode.value(),
                self.sl_dilate_x.value()
            )
            if preview_img is not None:
                self.show_image(preview_img, self.lbl_result)

    def show_image(self, img_source, label_widget):
        if isinstance(img_source, str):
            pixmap = QPixmap(img_source)
        else:
            # 🔴 修改点：兼容单通道二值化图与三通道彩色图
            if len(img_source.shape) == 2:  # 单通道（二值图）
                h, w = img_source.shape
                q_img = QImage(img_source.data, w, h, w, QImage.Format_Grayscale8)
            else:  # 三通道（RGB图）
                h, w, c = img_source.shape
                q_img = QImage(img_source.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _pulse_loading(self):
        self._badge_tick = (self._badge_tick + 1) % 3
        self.status_badge.setText(f"⬤  MODEL LOADING{'.' * (self._badge_tick + 1)}")

    def on_model_loaded(self):
        self._pulse_timer.stop();
        self.status_badge.setText("⬤  READY");
        self.status_badge.set_color("#3FB950")
        self.text_edit.append("\n// ✓ 深度学习模型装载完毕\n")

    def toggle_params(self):
        is_a = self.radio_a.isChecked()
        self.group_a_params.setEnabled(is_a);
        self.group_b_params.setEnabled(not is_a)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图像', '.', 'Images (*.jpg *.png *.bmp)')
        if fname:
            self.current_image_path = fname;
            self.lbl_path.setText(os.path.basename(fname))
            self.show_image(fname, self.lbl_origin);
            self.lbl_result.clear();
            self.text_edit.clear()

    def start_recognition(self):
        if not self.current_image_path: return QMessageBox.warning(self, "提示", "请先加载图片！")
        self.btn_run.setEnabled(False);
        self.btn_run.setText("⏳  PROCESSING...")
        self.worker = OCRWorker(
            'A' if self.radio_a.isChecked() else 'B', self.current_image_path,
            self.algo_a, self.algo_b, self.slider_conf.value() / 100.0,
            self.sl_erode.value(), self.sl_dilate_x.value(), 3, self.sl_unclip.value() / 10.0
        )
        self.worker.finished_signal.connect(self.update_ui);
        self.worker.start()

    def update_ui(self, txt, img, cost, err):
        self.btn_run.setEnabled(True);
        self.btn_run.setText("▶  RUN RECOGNITION")
        if err: return QMessageBox.critical(self, "错误", err)
        self.lbl_time.setText(f"⏱  {cost:.4f} s");
        self.text_edit.setText(txt)
        self.current_result_img = img;
        self.btn_export.setEnabled(True)
        if img is not None: self.show_image(img, self.lbl_result)

    def export_results(self):
        if self.current_result_img is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存", "OCR_Result", "Text (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f: f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "成功", "✅ 文本已导出！")


if __name__ == '__main__':
    app = QApplication(sys.argv);
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    window = OCRSystemUI();
    window.show();
    sys.exit(app.exec_())
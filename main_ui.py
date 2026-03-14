import sys
import os
import traceback

from PyQt5.QtCore import Qt

# 继续保持关闭双重放大，确保图片渲染的绝对清晰与尺寸可控
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QRadioButton, QTextEdit,
    QGroupBox, QMessageBox, QSlider, QDesktopWidget, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QThread, pyqtSignal

# 导入双引擎算法模块
from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB

# =========================================================
# 🎨 巨物化工业风 QSS (全面放大所有组件)
# =========================================================
MODERN_QSS = """
QMainWindow, QWidget { background-color: #181818; color: #E0E0E0; font-family: "Microsoft YaHei"; }
QGroupBox { 
    font-size: 18px; font-weight: bold; border: 1px solid #333333; 
    border-radius: 8px; margin-top: 25px; background-color: #202020; padding-top: 22px; padding-bottom: 10px;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; color: #58A6FF; }
QLabel { font-size: 18px; }
QLabel#lbl_nav_title { font-size: 28px; font-weight: bold; color: #FFFFFF; letter-spacing: 2px; }
QLabel#lbl_img_title { font-size: 16px; color: #8B949E; font-weight: bold; margin-bottom: 8px; }

QRadioButton { font-size: 18px; color: #C9D1D9; spacing: 12px; }
QRadioButton::indicator { width: 22px; height: 22px; border-radius: 11px; border: 2px solid #58A6FF; background-color: #181818; }
QRadioButton::indicator:checked { background-color: #58A6FF; }

QPushButton { 
    border: 1px solid #30363D; border-radius: 8px; padding: 12px; 
    background-color: #21262D; color: #C9D1D9; font-size: 18px; font-weight: bold;
}
QPushButton:hover { background-color: #30363D; }
QPushButton#btn_run_main { 
    background-color: #1F6FEB; border: none; color: #FFFFFF; font-size: 22px; padding: 18px; border-radius: 10px;
}
QPushButton#btn_run_main:hover { background-color: #388BFD; }
QPushButton#btn_export_main { color: #3FB950; border: 2px solid #238636; background: transparent; padding: 14px; }
QPushButton#btn_export_main:hover { background-color: #238636; color: white; }

QTextEdit { 
    border: 1px solid #30363D; border-radius: 8px; background-color: #0D1117; 
    font-size: 20px; color: #E6EDF3; padding: 15px; line-height: 1.6;
}
QSlider::groove:horizontal { border: none; background: #30363D; height: 8px; border-radius: 4px; }
QSlider::handle:horizontal { background: #58A6FF; width: 20px; height: 20px; margin: -6px 0; border-radius: 10px; }
"""


# =========================================================
# 🖼️ 自定义图片渲染类
# =========================================================
class ImageLabel(QLabel):
    def __init__(self, placeholder_text=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #0D1117; border: 1px dashed #30363D; border-radius: 8px; font-size: 18px;")
        self.setText(placeholder_text)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(250, 250)
        self._pixmap = None

    def set_custom_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.setText("")
        self.update()

    def clear_image(self, text=""):
        self._pixmap = None
        self.setText(text)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap:
            painter = QPainter(self)
            label_size = self.size()
            pix_size = self._pixmap.size()

            if pix_size.width() <= label_size.width() and pix_size.height() <= label_size.height():
                scaled_pix = self._pixmap
            else:
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                scaled_pix = self._pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            x = (self.width() - scaled_pix.width()) // 2
            y = (self.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)


# ── 工作线程 ─────────
class OCRWorker(QThread):
    finished_signal = pyqtSignal(str, object, float, str)

    def __init__(self, algo_type, img_path, algo_a, algo_b, threshold, erode_sz, dilate_x, unclip_ratio):
        super().__init__()
        self.algo_type, self.img_path = algo_type, img_path
        self.algo_a, self.algo_b = algo_a, algo_b
        self.threshold, self.erode_sz, self.dilate_x = threshold, erode_sz, dilate_x
        self.unclip_ratio = unclip_ratio

    def run(self):
        result_text, result_img, error_msg, cost_time = "", None, "", 0.0
        try:
            if self.algo_type == 'A':
                print("[DEBUG] 正在执行 AlgorithmA (OpenCV+Tesseract)...")
                result_text, result_img, cost_time = self.algo_a.detect_and_recognize(
                    self.img_path, conf_threshold=self.threshold,
                    erode_size=self.erode_sz, dilate_x=self.dilate_x, dilate_y=3)
            else:
                print("[DEBUG] 正在执行 AlgorithmB (PaddleOCR)...")
                result_text, result_img, cost_time = self.algo_b.detect_and_recognize(
                    self.img_path, conf_threshold=self.threshold, unclip_ratio=self.unclip_ratio)
        except Exception as e:
            error_msg = str(e)
            traceback.print_exc()
        self.finished_signal.emit(result_text, result_img, cost_time, error_msg)


# ── 滑块组件 ─────────
class LabeledSlider(QWidget):
    def __init__(self, label, min_val, max_val, default, formatter=None, parent=None):
        super().__init__(parent)
        self.formatter = formatter or (lambda v: str(v))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 8, 5, 8)  # 上下边距调宽
        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(100)  # 加宽标签区防止巨无霸字体挤压
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)
        self.val_lbl = QLabel(self.formatter(default))
        self.val_lbl.setMinimumWidth(50)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setStyleSheet("color: #58A6FF; font-weight: bold; font-size: 18px;")
        self.slider.valueChanged.connect(lambda v: self.val_lbl.setText(self.formatter(v)))
        layout.addWidget(self.lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)

    def value(self): return self.slider.value()


# ── 主界面 ─────────
class OCRSystemUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("桌面端可视化双引擎 OCR 系统")
        # 窗口全面扩大以容纳大字号
        self.resize(1400, 850)
        self.center_on_screen()
        self.setStyleSheet(MODERN_QSS)

        self.algo_a = AlgorithmA()
        self.algo_b = AlgorithmB()
        self.current_image_path = None
        self.current_result_img = None

        self.init_ui()

    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        # === 左侧控制台 ===
        sidebar = QWidget()
        sidebar.setFixedWidth(420)  # 史诗级加宽，彻底消灭文字拥挤
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("OCR 双引擎控制台")
        title.setObjectName("lbl_nav_title")
        side_layout.addWidget(title)
        side_layout.addSpacing(15)

        self.btn_load = QPushButton("📁 载入本地测试图像")
        self.btn_load.clicked.connect(self.load_image)
        side_layout.addWidget(self.btn_load)

        algo_box = QGroupBox("核心识别引擎选择")
        al = QVBoxLayout()
        al.setSpacing(15)
        self.radio_a = QRadioButton("方案 A: OpenCV 机器视觉")
        self.radio_b = QRadioButton("方案 B: PaddleOCR 深度学习")
        self.radio_b.setChecked(True)
        self.radio_a.toggled.connect(self.toggle_params)
        al.addWidget(self.radio_a)
        al.addWidget(self.radio_b)
        algo_box.setLayout(al)
        side_layout.addWidget(algo_box)

        self.slider_conf = LabeledSlider("过滤置信度", 10, 95, 50, lambda v: f"{v / 100:.2f}")
        side_layout.addWidget(self.slider_conf)

        self.group_a_params = QGroupBox("方案 A 专属: 形态学预处理")
        apl = QVBoxLayout()
        self.sl_dilate_x = LabeledSlider("横向膨胀度", 1, 40, 15)
        self.sl_erode = LabeledSlider("腐蚀去噪核", 1, 5, 2)
        self.sl_dilate_x.slider.sliderReleased.connect(self.update_morphology_preview)
        self.sl_erode.slider.sliderReleased.connect(self.update_morphology_preview)
        apl.addWidget(self.sl_dilate_x)
        apl.addWidget(self.sl_erode)
        self.group_a_params.setLayout(apl)
        side_layout.addWidget(self.group_a_params)

        self.group_b_params = QGroupBox("方案 B 专属: DBNET 检测")
        bpl = QVBoxLayout()
        self.sl_unclip = LabeledSlider("框膨胀率", 10, 30, 15, lambda v: f"{v / 10:.1f}")
        bpl.addWidget(self.sl_unclip)
        self.group_b_params.setLayout(bpl)
        side_layout.addWidget(self.group_b_params)

        side_layout.addStretch()

        self.lbl_time = QLabel("系统待命...")
        self.lbl_time.setStyleSheet("color: #8B949E; font-weight: bold; font-size: 16px;")
        side_layout.addWidget(self.lbl_time)
        side_layout.addSpacing(5)

        self.btn_run = QPushButton("▶ 启动端到端识别")
        self.btn_run.setObjectName("btn_run_main")
        self.btn_run.clicked.connect(self.start_recognition)
        side_layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("💾 导出当前文本结果")
        self.btn_export.setObjectName("btn_export_main")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        side_layout.addWidget(self.btn_export)
        main_layout.addWidget(sidebar)

        # === 右侧展示面板 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(25)

        # 图像展示区
        img_layout = QHBoxLayout()
        img_layout.setSpacing(25)

        box_ori = QVBoxLayout()
        lbl_t1 = QLabel("INPUT / 原始图像预览")
        lbl_t1.setObjectName("lbl_img_title")
        self.lbl_origin = ImageLabel("请在左侧载入图像")
        box_ori.addWidget(lbl_t1)
        box_ori.addWidget(self.lbl_origin)

        box_res = QVBoxLayout()
        lbl_t2 = QLabel("OUTPUT / 处理结果与特征渲染")
        lbl_t2.setObjectName("lbl_img_title")
        self.lbl_result = ImageLabel("等待处理...")
        box_res.addWidget(lbl_t2)
        box_res.addWidget(self.lbl_result)

        img_layout.addLayout(box_ori, stretch=1)
        img_layout.addLayout(box_res, stretch=1)
        right_layout.addLayout(img_layout, stretch=6)

        # 文本输出区
        box_txt = QVBoxLayout()
        lbl_t3 = QLabel("RECOGNITION OUTPUT / 结构化文本输出")
        lbl_t3.setObjectName("lbl_img_title")
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText(">> 识别结果将在此处输出...")
        box_txt.addWidget(lbl_t3)
        box_txt.addWidget(self.text_edit)

        right_layout.addLayout(box_txt, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)
        self.toggle_params()

    def toggle_params(self):
        is_a = self.radio_a.isChecked()
        self.group_a_params.setEnabled(is_a)
        self.group_b_params.setEnabled(not is_a)
        self.update_morphology_preview()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图像', '.', 'Images (*.jpg *.png *.bmp)')
        if fname:
            self.current_image_path = fname
            self.show_image(fname, self.lbl_origin)
            self.lbl_result.clear_image("等待处理...")
            self.text_edit.clear()
            self.lbl_time.setText("✅ 图像已载入")
            self.update_morphology_preview()

    def update_morphology_preview(self):
        if self.radio_a.isChecked() and self.current_image_path:
            try:
                preview_img = self.algo_a.get_morphology_preview(
                    self.current_image_path,
                    self.sl_erode.value(),
                    self.sl_dilate_x.value()
                )
                if preview_img is not None:
                    self.show_image(preview_img, self.lbl_result)
            except Exception as e:
                print(f"预览刷新异常: {e}")

    def show_image(self, img_source, custom_label):
        try:
            if isinstance(img_source, str):
                pixmap = QPixmap(img_source)
            else:
                if len(img_source.shape) == 2:
                    h, w = img_source.shape
                    q_img = QImage(img_source.data, w, h, w, QImage.Format_Grayscale8)
                else:
                    h, w, c = img_source.shape
                    q_img = QImage(img_source.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)

            custom_label.set_custom_pixmap(pixmap)
        except Exception as e:
            print(f"图像渲染失败: {e}")

    def start_recognition(self):
        if not self.current_image_path:
            return QMessageBox.warning(self, "提示", "请先载入需要识别的图像！")

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 引擎推理计算中...")
        self.lbl_time.setText("正在执行底层前向传播...")
        self.text_edit.clear()

        self.worker = OCRWorker(
            'A' if self.radio_a.isChecked() else 'B',
            self.current_image_path,
            self.algo_a, self.algo_b,
            self.slider_conf.value() / 100.0,
            self.sl_erode.value(),
            self.sl_dilate_x.value(),
            self.sl_unclip.value() / 10.0
        )
        self.worker.finished_signal.connect(self.update_ui)
        self.worker.start()

    def update_ui(self, txt, img, cost, err):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 启动端到端识别")

        if err:
            self.lbl_time.setText("❌ 系统发生异常")
            return QMessageBox.critical(self, "执行错误", f"引擎抛出异常：\n{err}")

        self.lbl_time.setText(f"✅ 推理完成 | 耗时: {cost:.2f} s")
        self.text_edit.setText(txt)
        self.current_result_img = img
        self.btn_export.setEnabled(True)

        if img is not None:
            self.show_image(img, self.lbl_result)

    def export_results(self):
        if not self.text_edit.toPlainText(): return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文本", "OCR_Result", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "成功", "文本已成功导出到本地！")


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = OCRSystemUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("====== 系统遇到致命错误中止 ======")
        traceback.print_exc()
        input("按任意键退出控制台...")
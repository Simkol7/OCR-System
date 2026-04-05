"""
main_ui.py — 桌面端双引擎OCR系统主界面入口
优化[优化-3]：为模块、类和核心方法补全文档字符串
"""
import sys
import os
import traceback

from PyQt5.QtCore import Qt

# 关闭系统级DPI双重缩放，确保图片渲染尺寸可控
if hasattr(Qt, "AA_EnableHighDpiScaling"):
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)

# 优化[优化-4]：移除已废弃的 QDesktopWidget，改用 QApplication.primaryScreen()
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QRadioButton, QTextEdit,
    QGroupBox, QMessageBox, QSlider, QSizePolicy, QApplication,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QThread, pyqtSignal

from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB
from common.utils import load_config

# =========================================================
# 全局样式表（深色工业风，适配高分辨率屏幕）
# =========================================================
MODERN_QSS = """
QMainWindow, QWidget { background-color: #181818; color: #E0E0E0; font-family: "Microsoft YaHei"; }
QGroupBox {
    font-size: 18px; font-weight: bold; border: 1px solid #333333;
    border-radius: 8px; margin-top: 25px; background-color: #202020;
    padding-top: 22px; padding-bottom: 10px;
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
    background-color: #1F6FEB; border: none; color: #FFFFFF; font-size: 22px;
    padding: 18px; border-radius: 10px;
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
# 自定义图片渲染控件
# =========================================================
class ImageLabel(QLabel):
    """
    支持等比例自适应缩放的图片展示控件。
    图像尺寸小于控件时原样显示；超出时等比缩放至控件范围内。
    """

    def __init__(self, placeholder_text=""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #0D1117; border: 1px dashed #30363D;"
            " border-radius: 8px; font-size: 18px;"
        )
        self.setText(placeholder_text)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(250, 250)
        self._pixmap = None

    def set_custom_pixmap(self, pixmap):
        """设置待显示的 QPixmap，并触发重绘。"""
        self._pixmap = pixmap
        self.setText("")
        self.update()

    def clear_image(self, text=""):
        """清除图像并显示占位文字。"""
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


# =========================================================
# OCR后台工作线程
# =========================================================
class OCRWorker(QThread):
    """
    在独立线程中运行OCR推理，通过 finished_signal 信号将结果返回主线程，
    避免长时间推理阻塞UI。
    信号参数：(识别文本: str, 结果图像: ndarray, 耗时: float, 错误信息: str)
    """
    finished_signal = pyqtSignal(str, object, float, str)

    def __init__(self, algo_type, img_path, algo_a, algo_b, threshold, erode_sz, dilate_x, unclip_ratio):
        super().__init__()
        self.algo_type = algo_type
        self.img_path = img_path
        self.algo_a = algo_a
        self.algo_b = algo_b
        self.threshold = threshold
        self.erode_sz = erode_sz
        self.dilate_x = dilate_x
        self.unclip_ratio = unclip_ratio

    def run(self):
        """线程入口：根据选择的引擎执行推理，异常时通过信号传回错误信息。"""
        result_text, result_img, error_msg, cost_time = "", None, "", 0.0
        try:
            if self.algo_type == "A":
                # dilate_y 从配置文件读取，不再硬编码为 3
                dilate_y = load_config().get("preprocess", {}).get("morph_dilate_y", 3)
                result_text, result_img, cost_time = self.algo_a.detect_and_recognize(
                    self.img_path,
                    conf_threshold=self.threshold,
                    erode_size=self.erode_sz,
                    dilate_x=self.dilate_x,
                    dilate_y=dilate_y,
                )
            else:
                result_text, result_img, cost_time = self.algo_b.detect_and_recognize(
                    self.img_path,
                    conf_threshold=self.threshold,
                    unclip_ratio=self.unclip_ratio,
                )
        except Exception as e:
            error_msg = str(e)
            traceback.print_exc()
        self.finished_signal.emit(result_text, result_img, cost_time, error_msg)


# =========================================================
# 形态学预览后台线程（普通-4：预览移至后台线程，避免主线程阻塞）
# =========================================================
class MorphologyPreviewWorker(QThread):
    """
    在后台执行形态学预处理预览，通过 preview_ready 信号将图像传回主线程渲染，
    防止大图预处理时短暂冻结界面。
    使用标志位协作退出，避免 QThread.terminate() 强制杀线程引发的资源泄漏。
    """
    preview_ready = pyqtSignal(object)

    def __init__(self, algo_a, img_path, erode_size, dilate_x):
        super().__init__()
        self.algo_a = algo_a
        self.img_path = img_path
        self.erode_size = erode_size
        self.dilate_x = dilate_x
        self._stop_requested = False  # 协作退出标志位

    def request_stop(self):
        """通知线程尽快退出（协作式，不强制杀线程）。"""
        self._stop_requested = True

    def run(self):
        """线程入口：执行预处理并发出预览图像信号。"""
        if self._stop_requested:
            return
        preview = self.algo_a.get_morphology_preview(self.img_path, self.erode_size, self.dilate_x)
        # 再次检查：防止预处理耗时期间外部已请求停止
        if not self._stop_requested and preview is not None:
            self.preview_ready.emit(preview)


# =========================================================
# 带数值标注的滑块组件
# =========================================================
class LabeledSlider(QWidget):
    """
    封装 QSlider，在右侧同步显示当前数值的复合控件，支持自定义格式化函数。
    """

    def __init__(self, label, min_val, max_val, default, formatter=None, parent=None):
        super().__init__(parent)
        self.formatter = formatter or (lambda v: str(v))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 8, 5, 8)

        self.lbl = QLabel(label)
        self.lbl.setMinimumWidth(100)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default)

        self.val_lbl = QLabel(self.formatter(default))
        self.val_lbl.setMinimumWidth(50)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setStyleSheet("color: #58A6FF; font-weight: bold; font-size: 18px;")

        # 滑块数值变化时同步更新显示标签
        self.slider.valueChanged.connect(lambda v: self.val_lbl.setText(self.formatter(v)))

        layout.addWidget(self.lbl)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_lbl)

    def value(self):
        return self.slider.value()


# =========================================================
# 主窗口
# =========================================================
class OCRSystemUI(QMainWindow):
    """
    双引擎OCR系统主窗口。
    提供图像载入、引擎切换（方案A/B）、参数调节、端到端识别、
    结果可视化展示及文本导出等完整交互功能。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("桌面端可视化双引擎 OCR 系统")
        self.resize(1400, 850)
        self.center_on_screen()
        self.setStyleSheet(MODERN_QSS)

        # 初始化双引擎实例
        self.algo_a = AlgorithmA()
        self.algo_b = AlgorithmB()
        self.current_image_path = None
        self.current_result_img = None

        self.init_ui()

    # ----------------------------------------------------------
    # 初始化方法
    # ----------------------------------------------------------

    def center_on_screen(self):
        """
        优化[优化-4]：将窗口居中显示。
        使用 QApplication.primaryScreen() 替代已废弃的 QDesktopWidget。
        """
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        """构建左侧控制台与右侧展示面板的整体布局，并绑定所有信号与槽。"""
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        # === 左侧控制台 ===
        sidebar = QWidget()
        sidebar.setFixedWidth(420)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("OCR 双引擎控制台")
        title.setObjectName("lbl_nav_title")
        side_layout.addWidget(title)
        side_layout.addSpacing(15)

        self.btn_load = QPushButton("载入本地测试图像")
        # 优化[优化-2]：槽函数命名统一采用 on_<控件>_<信号> 惯例
        self.btn_load.clicked.connect(self.on_btn_load_clicked)
        side_layout.addWidget(self.btn_load)

        # 引擎选择
        algo_box = QGroupBox("核心识别引擎选择")
        al = QVBoxLayout()
        al.setSpacing(15)
        self.radio_a = QRadioButton("方案 A: OpenCV 机器视觉")
        self.radio_b = QRadioButton("方案 B: PaddleOCR 深度学习")
        self.radio_b.setChecked(True)
        self.radio_a.toggled.connect(self.on_radio_engine_toggled)  # 优化[优化-2]
        al.addWidget(self.radio_a)
        al.addWidget(self.radio_b)
        algo_box.setLayout(al)
        side_layout.addWidget(algo_box)

        # 置信度滑块（两种方案共用）
        self.slider_conf = LabeledSlider("过滤置信度", 10, 95, 50, lambda v: f"{v / 100:.2f}")
        side_layout.addWidget(self.slider_conf)

        # 方案A专属参数
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

        # 方案B专属参数
        self.group_b_params = QGroupBox("方案 B 专属: DBNet 检测")
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
        self.btn_run.clicked.connect(self.on_btn_run_clicked)  # 优化[优化-2]
        side_layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("导出当前文本结果")
        self.btn_export.setObjectName("btn_export_main")
        self.btn_export.clicked.connect(self.on_btn_export_clicked)  # 优化[优化-2]
        self.btn_export.setEnabled(False)
        side_layout.addWidget(self.btn_export)
        main_layout.addWidget(sidebar)

        # === 右侧展示面板 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(25)

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

        # 初始化参数面板可用状态
        self.on_radio_engine_toggled()

    # ----------------------------------------------------------
    # 槽函数（优化[优化-2]：统一采用 on_<控件>_<信号> 命名惯例）
    # ----------------------------------------------------------

    def on_radio_engine_toggled(self, checked=None):
        """
        槽函数：引擎单选按钮切换时触发。
        根据当前选择启用/禁用对应方案的专属参数面板，并刷新预览。
        """
        is_a = self.radio_a.isChecked()
        self.group_a_params.setEnabled(is_a)
        self.group_b_params.setEnabled(not is_a)
        self.update_morphology_preview()

    def on_btn_load_clicked(self):
        """
        槽函数：点击"载入图像"按钮时触发。
        弹出文件选择对话框，载入图像后更新原图预览并刷新形态学预览。
        """
        fname, _ = QFileDialog.getOpenFileName(self, "选择图像", ".", "Images (*.jpg *.png *.bmp)")
        if fname:
            self.current_image_path = fname
            self.show_image(fname, self.lbl_origin)
            self.lbl_result.clear_image("等待处理...")
            self.text_edit.clear()
            self.lbl_time.setText("图像已载入")
            self.update_morphology_preview()

    def on_btn_run_clicked(self):
        """
        槽函数：点击"启动识别"按钮时触发。
        收集当前界面参数，启动 OCRWorker 后台线程执行推理。
        """
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先载入需要识别的图像！")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 引擎推理计算中...")
        self.lbl_time.setText("正在执行底层前向传播...")
        self.text_edit.clear()

        self.worker = OCRWorker(
            "A" if self.radio_a.isChecked() else "B",
            self.current_image_path,
            self.algo_a,
            self.algo_b,
            self.slider_conf.value() / 100.0,
            self.sl_erode.value(),
            self.sl_dilate_x.value(),
            self.sl_unclip.value() / 10.0,
        )
        # 优化[优化-2]：信号绑定到规范命名的槽函数
        self.worker.finished_signal.connect(self.on_ocr_worker_finished)
        self.worker.start()

    def on_ocr_worker_finished(self, txt, img, cost, err):
        """
        槽函数：OCRWorker 推理完成后触发。
        恢复按钮状态，显示识别结果文本与可视化图像。
        """
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ 启动端到端识别")

        if err:
            self.lbl_time.setText("系统发生异常")
            QMessageBox.critical(self, "执行错误", f"引擎抛出异常：\n{err}")
            return

        self.lbl_time.setText(f"推理完成 | 耗时: {cost:.2f} s")
        self.text_edit.setText(txt)
        self.current_result_img = img
        self.btn_export.setEnabled(True)

        if img is not None:
            self.show_image(img, self.lbl_result)

    def on_btn_export_clicked(self):
        """
        槽函数：点击"导出文本"按钮时触发。
        将识别结果写入用户选择的文件，写入失败时弹出错误提示。
        """
        if not self.text_edit.toPlainText():
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文本", "OCR_Result", "Text Files (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "成功", "文本已成功导出到本地！")
        except OSError as e:
            QMessageBox.critical(self, "导出失败", f"文件写入失败：{e}")

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def update_morphology_preview(self):
        """
        刷新方案A的形态学预处理预览图。
        运行于后台线程（MorphologyPreviewWorker），避免阻塞主线程。
        若上一次预览任务仍在运行，先终止再启动新任务。
        """
        if not (self.radio_a.isChecked() and self.current_image_path):
            return
        if hasattr(self, "_preview_worker") and self._preview_worker.isRunning():
            # 协作式退出：请求停止后等待线程自然结束，避免 terminate() 强杀资源泄漏
            self._preview_worker.request_stop()
            self._preview_worker.wait(500)
        self._preview_worker = MorphologyPreviewWorker(
            self.algo_a,
            self.current_image_path,
            self.sl_erode.value(),
            self.sl_dilate_x.value(),
        )
        self._preview_worker.preview_ready.connect(lambda img: self.show_image(img, self.lbl_result))
        self._preview_worker.start()

    def show_image(self, img_source, target_label):
        """
        将图像渲染到指定的 ImageLabel 控件中。
        :param img_source: 文件路径（str）或 numpy ndarray（灰度/BGR）
        :param target_label: 目标 ImageLabel 实例
        """
        try:
            if isinstance(img_source, str):
                pixmap = QPixmap(img_source)
            else:
                if len(img_source.shape) == 2:
                    h, w = img_source.shape
                    q_img = QImage(img_source.data, w, h, w, QImage.Format_Grayscale8)
                else:
                    h, w, _ = img_source.shape
                    q_img = QImage(img_source.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
            target_label.set_custom_pixmap(pixmap)
        except Exception as e:
            # 渲染失败通过状态栏告知用户，而非仅打印到控制台
            self.lbl_time.setText(f"图像渲染失败: {e}")

    def closeEvent(self, event):
        """
        关闭窗口时确保所有后台线程安全退出，防止资源泄漏。
        优先通过标志位协作退出；超时后才使用 terminate() 兜底强制终止。
        """
        for attr in ("worker", "_preview_worker"):
            t = getattr(self, attr, None)
            if t is None or not t.isRunning():
                continue
            # 发出协作退出信号（支持 request_stop 的线程）
            if hasattr(t, "request_stop"):
                t.request_stop()
            t.wait(2000)
            # 超时后兜底强制终止，防止窗口卡死
            if t.isRunning():
                t.terminate()
                t.wait(1000)
        event.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = OCRSystemUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("====== 系统遇到致命错误中止 ======")
        traceback.print_exc()
        input("按任意键退出控制台...")

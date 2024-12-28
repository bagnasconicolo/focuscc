import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QScrollArea, QSpinBox,
    QFormLayout, QFrame, QTextEdit, QLineEdit, QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
import pyqtgraph as pg

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Global variables
contrast = 1.0
brightness = 0
saturation = 1.0
black_point = 0
running = False
cap = None
threshold_value = 128.0
threshold_max_limit = 255.0
cooldown_time = 5  # in seconds
last_saved_time = 0
prefix = "event"
save_images = True


class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def run(self):
        global cap, running
        while running and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.new_frame.emit(frame)
            time.sleep(0.01)


class CharmCCM(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("charmCCM - Computer Vision for Cloud Chamber Events")
        self.setWindowIcon(QIcon("ccmicon.ico"))
        

        self.setGeometry(100, 100, 1200, 800)


        # We keep *all* x-values and mean_values (no pop).
        self.xvals = []
        self.mean_values = []

        self.threshold = threshold_value

        # Keep track of event lines & labels
        self.event_lines = []
        self.event_labels = []
        self.max_event_lines = 10

        self.camera_thread = None

        # Stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QLabel { color: #333; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaa;
                padding: 10px 10px;
                border-radius: 4px;
                margin-top: 6px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 10px;
                margin-bottom: 8px;
                background-color: #ddd;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
            QSlider::groove:horizontal { height: 6px; background: #ddd; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4CAF50; width: 14px; margin: -4px 0; border-radius: 7px; }
        """)

        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        content_frame = QFrame()
        content_layout = QHBoxLayout(content_frame)
        self.main_layout.addWidget(content_frame, stretch=1)

        # Video + Plot Section
        self.video_plot_layout = QVBoxLayout()

        # Video Label
        self.video_label = QLabel()
        self.video_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_plot_layout.addWidget(self.video_label)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("left", "Mean Intensity")
        self.plot_widget.setLabel("bottom", "Frame Count")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.enableAutoRange(axis='y', enable=True)

        # Add a black border to the plot (NEW LINE)
        self.plot_widget.setStyleSheet("border: 2px solid black;")

        self.plot_curve = self.plot_widget.plot([], pen=pg.mkPen("b", width=2))
        self.threshold_line = pg.InfiniteLine(
            pos=self.threshold,
            angle=0,
            pen=pg.mkPen("r", style=Qt.DashLine, width=2),
            movable=False
        )
        self.plot_widget.addItem(self.threshold_line)

        self.video_plot_layout.addWidget(self.plot_widget)
        content_layout.addLayout(self.video_plot_layout, stretch=2)

        # Controls Section
        self.controls_layout = QVBoxLayout()
        self.add_camera_controls()
        self.add_action_buttons()
        self.add_threshold_controls()
        self.add_prefix_and_cooldown_controls()
        self.add_log_box()
        self.controls_layout.addStretch()

        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_widget.setLayout(self.controls_layout)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(300)
        content_layout.addWidget(self.scroll_area, stretch=1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ----------------------------------------------------------------
    #  UI Setup
    # ----------------------------------------------------------------
    def add_camera_controls(self):
        control_group = QGroupBox("Camera Controls")
        group_layout = QVBoxLayout()

        self.add_slider("Contrast", 0.5, 3.0, 0.1, contrast, self.update_contrast, group_layout)
        self.add_slider("Brightness", -100, 100, 1, brightness, self.update_brightness, group_layout)
        self.add_slider("Saturation", 0.5, 3.0, 0.1, saturation, self.update_saturation, group_layout)
        self.add_slider("Black Point", -100, 100, 1, black_point, self.update_black_point, group_layout)

        control_group.setLayout(group_layout)
        self.controls_layout.addWidget(control_group)

    def add_action_buttons(self):
        action_group = QGroupBox("Actions")
        group_layout = QVBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        group_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setStyleSheet("background-color: red; color: white; border: none; padding: 6px; border-radius: 4px;")  # NEW LINE
        self.stop_button.clicked.connect(self.stop_camera)
        group_layout.addWidget(self.stop_button)

        self.save_checkbox = QCheckBox("Enable Image Saving")
        self.save_checkbox.setChecked(True)
        self.save_checkbox.stateChanged.connect(self.toggle_image_saving)
        group_layout.addWidget(self.save_checkbox)

        action_group.setLayout(group_layout)
        self.controls_layout.addWidget(action_group)

    def add_threshold_controls(self):
        threshold_group = QGroupBox("Threshold Control")
        group_layout = QVBoxLayout()

        slider_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, int(threshold_max_limit * 10))
        self.threshold_slider.setValue(int(self.threshold * 10))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        slider_layout.addWidget(self.threshold_slider)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0, threshold_max_limit)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setValue(self.threshold)
        self.threshold_spinbox.valueChanged.connect(self.spinbox_threshold_changed)
        slider_layout.addWidget(self.threshold_spinbox)

        group_layout.addLayout(slider_layout)
        self.threshold_value_label = QLabel(f"Threshold Value: {self.threshold:.1f}")
        group_layout.addWidget(self.threshold_value_label)

        threshold_group.setLayout(group_layout)
        self.controls_layout.addWidget(threshold_group)

    def add_prefix_and_cooldown_controls(self):
        prefix_group = QGroupBox("Prefix & Cooldown")
        form_layout = QFormLayout()

        self.prefix_edit = QLineEdit(prefix)
        self.prefix_edit.textChanged.connect(self.update_prefix)
        form_layout.addRow("Filename Prefix:", self.prefix_edit)

        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(cooldown_time)
        self.cooldown_spinbox.valueChanged.connect(self.update_cooldown)
        form_layout.addRow("Cooldown Time (s):", self.cooldown_spinbox)

        prefix_group.setLayout(form_layout)
        self.controls_layout.addWidget(prefix_group)

    def add_log_box(self):
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: black; color: lime;")
        log_layout.addWidget(self.log_box)

        log_group.setLayout(log_layout)
        self.controls_layout.addWidget(log_group)

    def add_slider(self, label, min_val, max_val, step, default_val, callback, layout):
        slider_label = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(default_val / step))
        slider.setSingleStep(1)
        slider.valueChanged.connect(lambda val: callback(val * step))
        layout.addWidget(slider_label)
        layout.addWidget(slider)

    # ----------------------------------------------------------------
    #  Logging & Misc
    # ----------------------------------------------------------------
    def log_message(self, message):
        self.log_box.append(f"{time.strftime('%H:%M:%S')} - {message}")

    def update_prefix(self, text):
        global prefix
        prefix = text
        self.log_message(f"Filename prefix updated to '{prefix}'.")

    def update_cooldown(self, value):
        global cooldown_time
        cooldown_time = value
        self.log_message(f"Cooldown time updated to {cooldown_time} seconds.")

    # ----------------------------------------------------------------
    #  Plot Updates (Scrolling)
    # ----------------------------------------------------------------
    def update_plot(self):
        if not self.xvals:
            return

        self.plot_curve.setData(self.xvals, self.mean_values)
        self.threshold_line.setValue(self.threshold)

        n = len(self.xvals)
        if n < 300:
            self.plot_widget.setXRange(0, 300, padding=0)
        else:
            xmin = self.xvals[-300]
            xmax = self.xvals[-1]
            self.plot_widget.setXRange(xmin, xmax, padding=0)

    # ----------------------------------------------------------------
    #  Camera Actions
    # ----------------------------------------------------------------
    def start_camera(self):
        global cap, running
        if running:
            self.log_message("Camera is already running.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_message("Error: Unable to access the camera.")
            return

        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        running = True
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.handle_new_frame)
        self.camera_thread.start()

        self.timer.start(50)
        self.log_message("Camera started.")

    def stop_camera(self):
        global cap, running
        running = False

        if self.camera_thread:
            self.camera_thread.wait()
            self.camera_thread = None

        if cap and cap.isOpened():
            cap.release()
            cap = None

        self.timer.stop()
        self.video_label.clear()
        self.log_message("Camera stopped.")

    # ----------------------------------------------------------------
    #  Frame Handling
    # ----------------------------------------------------------------
    def handle_new_frame(self, frame: np.ndarray):
        global last_saved_time

        # Adjust brightness/contrast/black point
        if brightness != 0 or black_point != 0 or contrast != 1.0:
            table = np.arange(256, dtype=np.float32)
            shift = brightness - black_point
            scale = contrast
            table = scale * (table + shift)
            table = np.clip(table, 0, 255).astype(np.uint8)
            frame = cv2.LUT(frame, table)

        # Saturation
        if saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            hsv = hsv.astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Gray / Mean
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)

        # Append new data
        if not self.xvals:
            self.xvals.append(0)
        else:
            self.xvals.append(self.xvals[-1] + 1)
        self.mean_values.append(mean_val)

        self.update_plot()

        # Check threshold => save
        current_time = time.time()
        if mean_val > self.threshold:
            if current_time - last_saved_time >= cooldown_time:
                last_saved_time = current_time
                if save_images:
                    filename = f"{prefix}_{int(current_time)}.png"
                    cv2.imwrite(filename, frame)
                    self.log_message(f"Event detected and image saved as '{filename}'.")

                    x_event = self.xvals[-1]
                    marker = pg.InfiniteLine(
                        pos=x_event,
                        angle=90,
                        pen=pg.mkPen(color='lime', width=2),
                        movable=False
                    )
                    self.plot_widget.addItem(marker)
                    self.event_lines.append(marker)

                    text_item = pg.TextItem(filename, color='green', anchor=(0, 1))
                    local_max = max(self.mean_values[-10:]) if len(self.mean_values) >= 10 else max(self.mean_values)
                    y_pos = local_max * 1.1 if local_max > 0 else (self.threshold + 10)
                    text_item.setPos(x_event, y_pos)
                    self.plot_widget.addItem(text_item)
                    self.event_labels.append(text_item)

                    # Prune old markers
                    if len(self.event_lines) > self.max_event_lines:
                        old_line = self.event_lines.pop(0)
                        self.plot_widget.removeItem(old_line)
                    if len(self.event_labels) > self.max_event_lines:
                        old_label = self.event_labels.pop(0)
                        self.plot_widget.removeItem(old_label)

        # Display processed frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def toggle_image_saving(self, state):
        global save_images
        save_images = (state == Qt.Checked)
        self.log_message(f"Image saving {'enabled' if save_images else 'disabled'}.")

    def update_contrast(self, value):
        global contrast
        contrast = value
        self.log_message(f"Contrast updated to {contrast:.2f}.")

    def update_brightness(self, value):
        global brightness
        brightness = int(value)
        self.log_message(f"Brightness updated to {brightness}.")

    def update_saturation(self, value):
        global saturation
        saturation = value
        self.log_message(f"Saturation updated to {saturation:.2f}.")

    def update_black_point(self, value):
        global black_point
        black_point = int(value)
        self.log_message(f"Black point updated to {black_point}.")

    def update_threshold(self, value):
        self.threshold = value / 10.0
        self.threshold_spinbox.setValue(self.threshold)
        self.threshold_value_label.setText(f"Threshold Value: {self.threshold:.1f}")
        self.log_message(f"Threshold updated to {self.threshold:.1f}.")

    def spinbox_threshold_changed(self, value):
        self.threshold_slider.setValue(int(value * 10))
        self.update_threshold(int(value * 10))

    def update_frame(self):
        pass

    def closeEvent(self, event):
        global cap, running
        running = False

        if self.camera_thread:
            self.camera_thread.wait()
            self.camera_thread = None

        if cap and cap.isOpened():
            cap.release()

        self.timer.stop()
        self.log_message("Application closed.")
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('ccmicon.png'))

    window = CharmCCM()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

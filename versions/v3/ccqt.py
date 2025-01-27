import sys
import time
import json
import cv2
import numpy as np
import subprocess
import platform
import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QTabWidget, QSplitter,
    QPlainTextEdit, QLineEdit, QCheckBox, QComboBox,
    QGridLayout, QMessageBox, QStyleFactory, QStyle, QFrame,
    QDialog, QDialogButtonBox, QStackedWidget, QRadioButton, QButtonGroup,
    QScrollBar, QToolTip, QSpacerItem, QSizePolicy
)

### Global variables
FRAME_WIDTH = 960
FRAME_HEIGHT = 640
# When stacked vertically => final stacked array: (2*FRAME_HEIGHT) x FRAME_WIDTH
# Aspect ratio ~ 960 : 1280 => 0.75
DEFAULT_CAMERA = 0
THRESHOLD_MIN = 100
THRESHOLD_MAX = 5000
CANNY_LOW_MIN = 10
CANNY_LOW_MAX = 300
CANNY_HIGH_MIN = 50
CANNY_HIGH_MAX = 300

running = False
save_events = False
only_view = True
threshold = 1000
low_threshold = 50
high_threshold = 150
cap = None
selected_camera = DEFAULT_CAMERA
filename_prefix = "event"
cooldown = 5  # seconds
last_event_time = 0
cooldown_enabled = True
contrast = 1.0
brightness = 0
saturation = 1.0
black_point = 0
preset_name = "default"
detection_area = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]  # [x1, y1, x2, y2]
logs = []
gaussian_blur = 5
canny_low_min = CANNY_LOW_MIN
canny_low_max = CANNY_LOW_MAX
canny_high_min = CANNY_HIGH_MIN
canny_high_max = CANNY_HIGH_MAX

def list_cameras():
    """Lists available cameras by probing indices and getting device names."""
    available_cameras = []
    for i in range(10):  # Check the first 10 indices
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            device_name = get_camera_name(i)
            available_cameras.append((i, device_name))
            test_cap.release()
    return available_cameras

def get_camera_name(index):
    """Gets the camera name using system_profiler on macOS or v4l2-ctl on Linux."""
    if platform.system() == "Darwin":  # macOS
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'],
                                    capture_output=True, text=True)
            lines = result.stdout.split('\n')
            camera_name = None
            for i, line in enumerate(lines):
                if 'Model ID' in line:
                    camera_name = lines[i - 1].strip()
                if 'Unique ID' in line and camera_name:
                    return camera_name
        except Exception:
            return f"Camera {index}"
    else:  # Linux or others
        try:
            result = subprocess.run(['v4l2-ctl', '--device', f'/dev/video{index}', '--info'],
                                    capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Card type' in line:
                    return line.split(':')[1].strip()
        except Exception:
            return f"Camera {index}"
    return f"Camera {index}"

class FocusCC(QMainWindow):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusCC - Proportional Feeds")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # We'll store the "stacked" BGR image in self.last_stacked_frame
        self.last_stacked_frame = None

        splitter_main = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(splitter_main, stretch=1)

        # Left side: video
        self.video_widget = QWidget()
        self.video_layout = QVBoxLayout(self.video_widget)
        self.video_widget.setLayout(self.video_layout)
        splitter_main.addWidget(self.video_widget)

        self.video_label = QLabel()
        # Let the label expand
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)  # center the pixmap if there's blank space
        self.video_layout.addWidget(self.video_label)

        # Right side: vertical splitter for (tabs + log)
        right_splitter = QSplitter(Qt.Vertical)
        splitter_main.addWidget(right_splitter)

        top_right_container = QWidget()
        top_right_layout = QVBoxLayout(top_right_container)
        top_right_container.setLayout(top_right_layout)
        right_splitter.addWidget(top_right_container)

        self.title_label = QLabel("FocusCC - Python CV for Cloud Chambers")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin: 6px 0;")
        top_right_layout.addWidget(self.title_label, 0, Qt.AlignHCenter)

        self.tab_widget = QTabWidget()
        top_right_layout.addWidget(self.tab_widget, stretch=1)

        # Build separate "Main" and "Events" tab
        self.build_tab_main()
        self.build_tab_events()

        # Merge the old "Video" + "Preset" + "Gaussian Blur" into one tab
        self.build_tab_videopreset()

        # "Advanced" tab now only has canny min/max entries, since we moved the blur slider
        self.build_tab_advanced()

        # Build log area
        self.build_log_frame()
        right_splitter.addWidget(self.log_frame)

        # List cameras
        self.camera_list = list_cameras()
        self.populate_camera_combo()

        # Timer for frames
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_feed)

        self.load_initial_values()

    # -------------------------------------------------------------------------
    #  MAIN TAB
    # -------------------------------------------------------------------------
    def build_tab_main(self):
        self.tab_main = QWidget()
        layout = QVBoxLayout(self.tab_main)

        group_threshold = QGroupBox("Threshold and Canny")
        group_layout = QVBoxLayout(group_threshold)
        layout.addWidget(group_threshold)

        thresh_label = QLabel("Event Threshold")
        group_layout.addWidget(thresh_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(THRESHOLD_MIN, THRESHOLD_MAX)
        self.threshold_slider.setValue(threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        group_layout.addWidget(self.threshold_slider)

        canny_low_label = QLabel("Canny Threshold (Low)")
        group_layout.addWidget(canny_low_label)
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(CANNY_LOW_MIN, CANNY_LOW_MAX)
        self.canny_low_slider.setValue(low_threshold)
        self.canny_low_slider.valueChanged.connect(self.update_canny_low)
        group_layout.addWidget(self.canny_low_slider)

        canny_high_label = QLabel("Canny Threshold (High)")
        group_layout.addWidget(canny_high_label)
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(CANNY_HIGH_MIN, CANNY_HIGH_MAX)
        self.canny_high_slider.setValue(high_threshold)
        self.canny_high_slider.valueChanged.connect(self.update_canny_high)
        group_layout.addWidget(self.canny_high_slider)

        group_area = QGroupBox("Detection Area")
        area_layout = QVBoxLayout(group_area)
        layout.addWidget(group_area)

        x1_label = QLabel("X1")
        area_layout.addWidget(x1_label)
        self.detection_area_x1_slider = QSlider(Qt.Horizontal)
        self.detection_area_x1_slider.setRange(0, FRAME_WIDTH)
        self.detection_area_x1_slider.setValue(detection_area[0])
        self.detection_area_x1_slider.valueChanged.connect(self.update_detection_area_x1)
        area_layout.addWidget(self.detection_area_x1_slider)

        y1_label = QLabel("Y1")
        area_layout.addWidget(y1_label)
        self.detection_area_y1_slider = QSlider(Qt.Horizontal)
        self.detection_area_y1_slider.setRange(0, FRAME_HEIGHT)
        self.detection_area_y1_slider.setValue(detection_area[1])
        self.detection_area_y1_slider.valueChanged.connect(self.update_detection_area_y1)
        area_layout.addWidget(self.detection_area_y1_slider)

        x2_label = QLabel("X2")
        area_layout.addWidget(x2_label)
        self.detection_area_x2_slider = QSlider(Qt.Horizontal)
        self.detection_area_x2_slider.setRange(0, FRAME_WIDTH)
        self.detection_area_x2_slider.setValue(detection_area[2])
        self.detection_area_x2_slider.valueChanged.connect(self.update_detection_area_x2)
        area_layout.addWidget(self.detection_area_x2_slider)

        y2_label = QLabel("Y2")
        area_layout.addWidget(y2_label)
        self.detection_area_y2_slider = QSlider(Qt.Horizontal)
        self.detection_area_y2_slider.setRange(0, FRAME_HEIGHT)
        self.detection_area_y2_slider.setValue(detection_area[3])
        self.detection_area_y2_slider.valueChanged.connect(self.update_detection_area_y2)
        area_layout.addWidget(self.detection_area_y2_slider)

        group_controls = QGroupBox("Camera / Event Control")
        ctrl_layout = QHBoxLayout(group_controls)
        layout.addWidget(group_controls)

        start_btn = QPushButton("Start Webcam")
        start_btn.clicked.connect(self.start_camera)
        ctrl_layout.addWidget(start_btn)

        stop_btn = QPushButton("Stop Webcam")
        stop_btn.clicked.connect(self.stop_camera)
        ctrl_layout.addWidget(stop_btn)

        self.save_check = QCheckBox("Save Events")
        self.save_check.setChecked(False)
        self.save_check.stateChanged.connect(self.toggle_save)
        ctrl_layout.addWidget(self.save_check)

        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.select_camera)
        ctrl_layout.addWidget(self.camera_combo)

        layout.addStretch()
        self.tab_widget.addTab(self.tab_main, "Main")

    # -------------------------------------------------------------------------
    #  EVENTS TAB (add a file list of the current folder)
    # -------------------------------------------------------------------------
    def build_tab_events(self):
        self.tab_events = QWidget()
        layout = QVBoxLayout(self.tab_events)

        prefix_label = QLabel("Filename Prefix")
        layout.addWidget(prefix_label)
        self.filename_entry = QLineEdit(filename_prefix)
        self.filename_entry.textChanged.connect(self.update_filename_prefix)
        layout.addWidget(self.filename_entry)

        cooldown_label = QLabel("Cooldown (seconds)")
        layout.addWidget(cooldown_label)
        self.cooldown_slider = QSlider(Qt.Horizontal)
        self.cooldown_slider.setRange(1, 60)
        self.cooldown_slider.setValue(cooldown)
        self.cooldown_slider.valueChanged.connect(self.update_cooldown)
        layout.addWidget(self.cooldown_slider)

        self.cooldown_check = QCheckBox("Enable Cooldown")
        self.cooldown_check.setChecked(True)
        self.cooldown_check.stateChanged.connect(self.toggle_cooldown)
        layout.addWidget(self.cooldown_check)

        # Add a group to display the files in the current folder
        files_group = QGroupBox("Files in Current Folder")
        files_layout = QVBoxLayout(files_group)
        layout.addWidget(files_group)

        self.file_list_combo = QComboBox()
        self.refresh_file_list()
        files_layout.addWidget(self.file_list_combo)

        layout.addStretch()
        self.tab_widget.addTab(self.tab_events, "Events")

    def refresh_file_list(self):
        """Populates self.file_list_combo with all files in the current directory (alphabetically)."""
        self.file_list_combo.clear()
        all_files = sorted([f for f in os.listdir('.') if os.path.isfile(f)])
        for f in all_files:
            self.file_list_combo.addItem(f)

    # -------------------------------------------------------------------------
    #  MERGED TAB: "VIDEO & PRESETS" + GAUSSIAN BLUR
    # -------------------------------------------------------------------------
    def build_tab_videopreset(self):
        """Merged content of the old 'Video' tab + 'Preset' tab + Gaussian Blur slider."""
        self.tab_videopreset = QWidget()
        layout = QVBoxLayout(self.tab_videopreset)

        # Video Controls
        video_group = QGroupBox("Video Controls")
        video_layout = QGridLayout(video_group)
        layout.addWidget(video_group)

        # Contrast
        contrast_label = QLabel("Contrast (0.5..3.0)")
        video_layout.addWidget(contrast_label, 0, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(5, 30)  # scaled by 10
        self.contrast_slider.setValue(int(contrast * 10))
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        video_layout.addWidget(self.contrast_slider, 1, 0)

        # Brightness
        brightness_label = QLabel("Brightness (-100..100)")
        video_layout.addWidget(brightness_label, 0, 1)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(brightness)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        video_layout.addWidget(self.brightness_slider, 1, 1)

        # Saturation
        sat_label = QLabel("Saturation (0.5..3.0)")
        video_layout.addWidget(sat_label, 2, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(5, 30)  # scaled by 10
        self.saturation_slider.setValue(int(saturation * 10))
        self.saturation_slider.valueChanged.connect(self.update_saturation)
        video_layout.addWidget(self.saturation_slider, 3, 0)

        # Black Point
        black_label = QLabel("Black Point (-100..100)")
        video_layout.addWidget(black_label, 2, 1)
        self.black_point_slider = QSlider(Qt.Horizontal)
        self.black_point_slider.setRange(-100, 100)
        self.black_point_slider.setValue(black_point)
        self.black_point_slider.valueChanged.connect(self.update_black_point)
        video_layout.addWidget(self.black_point_slider, 3, 1)

        # Gaussian Blur
        blur_label = QLabel("Gaussian Blur (odd values)")
        video_layout.addWidget(blur_label, 4, 0)
        self.gaussian_blur_slider = QSlider(Qt.Horizontal)
        self.gaussian_blur_slider.setRange(1, 15)
        self.gaussian_blur_slider.setValue(gaussian_blur)
        self.gaussian_blur_slider.valueChanged.connect(self.update_gaussian_blur)
        video_layout.addWidget(self.gaussian_blur_slider, 5, 0)

        # Reset button
        reset_btn = QPushButton("Reset Video Controls")
        reset_btn.clicked.connect(self.reset_video_controls)
        video_layout.addWidget(reset_btn, 5, 1)

        # Preset Group
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        layout.addWidget(preset_group)

        label = QLabel("Preset Name")
        preset_layout.addWidget(label)
        self.preset_name_entry = QLineEdit(preset_name)
        self.preset_name_entry.textChanged.connect(self.update_preset_name)
        preset_layout.addWidget(self.preset_name_entry)

        self.available_presets = [
            f.split(".json")[0] for f in os.listdir() if f.endswith(".json")
        ]
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.available_presets)
        preset_layout.addWidget(self.preset_combo)

        save_preset_btn = QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_preset_btn)

        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_preset)
        preset_layout.addWidget(load_preset_btn)

        layout.addStretch()
        self.tab_widget.addTab(self.tab_videopreset, "Video & Presets")

    # -------------------------------------------------------------------------
    #  ADVANCED TAB (now only has canny min/max, because blur moved)
    # -------------------------------------------------------------------------
    def build_tab_advanced(self):
        self.tab_advanced = QWidget()
        layout = QVBoxLayout(self.tab_advanced)

        canny_low_min_label = QLabel("Canny Low Min")
        layout.addWidget(canny_low_min_label)
        self.canny_low_min_entry = QLineEdit(str(canny_low_min))
        self.canny_low_min_entry.textChanged.connect(self.update_canny_low_min)
        layout.addWidget(self.canny_low_min_entry)

        canny_low_max_label = QLabel("Canny Low Max")
        layout.addWidget(canny_low_max_label)
        self.canny_low_max_entry = QLineEdit(str(canny_low_max))
        self.canny_low_max_entry.textChanged.connect(self.update_canny_low_max)
        layout.addWidget(self.canny_low_max_entry)

        canny_high_min_label = QLabel("Canny High Min")
        layout.addWidget(canny_high_min_label)
        self.canny_high_min_entry = QLineEdit(str(canny_high_min))
        self.canny_high_min_entry.textChanged.connect(self.update_canny_high_min)
        layout.addWidget(self.canny_high_min_entry)

        canny_high_max_label = QLabel("Canny High Max")
        layout.addWidget(canny_high_max_label)
        self.canny_high_max_entry = QLineEdit(str(canny_high_max))
        self.canny_high_max_entry.textChanged.connect(self.update_canny_high_max)
        layout.addWidget(self.canny_high_max_entry)

        layout.addStretch()
        self.tab_widget.addTab(self.tab_advanced, "Advanced")

    # -------------------------------------------------------------------------
    #  LOG AREA
    # -------------------------------------------------------------------------
    def build_log_frame(self):
        self.log_frame = QFrame()
        log_layout = QVBoxLayout(self.log_frame)

        label = QLabel("Event Log")
        log_layout.addWidget(label)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #000; color: #0f0; font-family: Courier;")
        log_layout.addWidget(self.log_box, stretch=1)

        filter_layout = QHBoxLayout()
        info_btn = QPushButton("Info")
        info_btn.clicked.connect(lambda: self.filter_logs("INFO"))
        filter_layout.addWidget(info_btn)

        warn_btn = QPushButton("Warning")
        warn_btn.clicked.connect(lambda: self.filter_logs("WARNING"))
        filter_layout.addWidget(warn_btn)

        err_btn = QPushButton("Error")
        err_btn.clicked.connect(lambda: self.filter_logs("ERROR"))
        filter_layout.addWidget(err_btn)

        log_layout.addLayout(filter_layout)

    # -------------------------------------------------------------------------
    #  CAMERA & FEED LOGIC
    # -------------------------------------------------------------------------
    def populate_camera_combo(self):
        self.camera_combo.clear()
        for idx, name in self.camera_list:
            self.camera_combo.addItem(f"{idx} - {name}")

    def load_initial_values(self):
        pass

    def start_camera(self):
        global cap, running
        if cap and cap.isOpened():
            self.log_message("Webcam is already active.")
            return

        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            self.log_message("Error: Unable to access the webcam.", "error")
            return

        running = True
        self.log_message(f"Webcam {selected_camera} started.")
        self.update_timer.start(10)

    def stop_camera(self):
        global running, cap
        running = False
        if cap:
            cap.release()
            cap = None
        self.log_message("Video feed stopped.")
        self.update_timer.stop()

    def select_camera(self, index):
        global selected_camera, running
        was_running = running
        if running:
            self.stop_camera()
        combo_text = self.camera_combo.currentText()
        cam_idx_str = combo_text.split(' - ')[0]
        selected_camera = int(cam_idx_str)
        self.log_message(f"Selected camera: {selected_camera}")
        if was_running:
            self.start_camera()

    def update_feed(self):
        global cap, running
        if not running or not cap or not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            self.log_message("Error: Unable to read the video feed.", "error")
            return

        # Always scale to (FRAME_WIDTH, FRAME_HEIGHT)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = self.apply_video_controls(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gb_val = gaussian_blur if (gaussian_blur % 2 == 1) else (gaussian_blur + 1)
        blurred = cv2.GaussianBlur(gray, (gb_val, gb_val), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # restrict edges
        x1, y1, x2, y2 = detection_area
        edges[:y1, :] = 0
        edges[y2:, :] = 0
        edges[:, :x1] = 0
        edges[:, x2:] = 0

        # check event
        event_detected = (np.sum(edges > 0) > threshold)
        current_time = time.time()
        if event_detected and (not cooldown_enabled or (current_time - last_event_time) > cooldown):
            self.handle_event(frame)
            self.update_last_event_time(current_time)

        self.stack_and_show(frame, edges)

    def apply_video_controls(self, frame):
        """
        Applies black_point, brightness, and contrast via a LUT,
        then handles saturation in HSV space.
        """
        global contrast, brightness, saturation, black_point

        # 1) Build a LUT for values 0..255 (float32 to avoid overflow)
        table = np.arange(256, dtype=np.float32)

        # 2) Combine brightness & black_point => shift
        shift = brightness - black_point

        # 3) Contrast is scale
        scale = contrast

        # 4) LUT: out = scale * (in + shift)
        table = scale * (table + shift)

        # 5) Clip to [0..255], convert to uint8
        table = np.clip(table, 0, 255).astype(np.uint8)

        # 6) Apply LUT
        frame = cv2.LUT(frame, table)

        # 7) Saturation in HSV
        if saturation != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
            hsv = hsv.astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    def handle_event(self, frame):
        global save_events
        if save_events:
            filename = f"{filename_prefix}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            self.log_message(f"Event detected and saved as {filename}")
        else:
            self.log_message("Event detected, no photo saved.")

        cv2.putText(frame, "EVENT DETECTED!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def update_last_event_time(self, t):
        global last_event_time
        last_event_time = t

    def stack_and_show(self, frame_bgr, edges):
        """
        Stacks 'frame_bgr' on top, 'edges' (converted to BGR) on bottom.
        Store the result in self.last_stacked_frame.
        Then call update_display_label().
        """
        # draw detection rectangle
        x1, y1, x2, y2 = detection_area
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        stacked = np.vstack((frame_bgr, edges_bgr))

        self.last_stacked_frame = stacked
        self.update_display_label()

    def update_display_label(self):
        if self.last_stacked_frame is None:
            return

        # Convert BGR -> RGB
        stacked_rgb = cv2.cvtColor(self.last_stacked_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = stacked_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(stacked_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        aspect_ratio = w / float(h)
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        container_ratio = label_w / float(label_h)

        if container_ratio > aspect_ratio:
            final_h = label_h
            final_w = int(aspect_ratio * final_h)
        else:
            final_w = label_w
            final_h = int(final_w / aspect_ratio)

        scaled_pixmap = pixmap.scaled(final_w, final_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a new QPixmap the size of the label, fill it with black
        letterbox_pixmap = QPixmap(label_w, label_h)
        letterbox_pixmap.fill(Qt.black)

        painter = QPainter(letterbox_pixmap)
        offset_x = (label_w - final_w) // 2
        offset_y = (label_h - final_h) // 2
        painter.drawPixmap(offset_x, offset_y, scaled_pixmap)
        painter.end()

        self.video_label.setPixmap(letterbox_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # re-draw the feed at the new label size
        self.update_display_label()

    # LOGGING
    def log_message(self, msg, level="info"):
        """
        Add a timestamp + level prefix to every log.
        """
        global logs
        timestamp = time.strftime("[%H:%M:%S]")
        level_str = level.upper()
        line = f"{timestamp} {level_str}: {msg}"
        logs.append(line + "\n")
        self.log_box.appendPlainText(line)

    def filter_logs(self, level):
        self.log_box.clear()
        level_up = level.upper()
        for line in logs:
            if f"{level_up}:" in line:
                self.log_box.appendPlainText(line.rstrip())

    # HANDLERS
    def toggle_save(self, state):
        global save_events, only_view
        save_events = (state == Qt.Checked)
        only_view = not save_events
        self.log_message(f"Save events: {'Enabled' if save_events else 'Disabled'}")

    def update_threshold(self, value):
        global threshold
        threshold = value
        self.log_message(f"Threshold updated to: {threshold}")

    def update_canny_low(self, value):
        global low_threshold
        low_threshold = value
        self.log_message(f"Canny low threshold updated to: {low_threshold}")

    def update_canny_high(self, value):
        global high_threshold
        high_threshold = value
        self.log_message(f"Canny high threshold updated to: {high_threshold}")

    def update_filename_prefix(self, text):
        global filename_prefix
        filename_prefix = text
        self.log_message(f"Filename prefix updated to: {filename_prefix}")

    def update_cooldown(self, value):
        global cooldown
        cooldown = value
        self.log_message(f"Cooldown updated to: {cooldown} seconds")

    def toggle_cooldown(self, state):
        global cooldown_enabled
        cooldown_enabled = (state == Qt.Checked)
        self.log_message(f"Cooldown: {'Enabled' if cooldown_enabled else 'Disabled'}")

    def update_contrast(self, slider_value):
        global contrast
        contrast = slider_value / 10.0
        self.log_message(f"Contrast updated to: {contrast:.2f}")

    def update_brightness(self, slider_value):
        global brightness
        brightness = slider_value
        self.log_message(f"Brightness updated to: {brightness}")

    def update_saturation(self, slider_value):
        global saturation
        saturation = slider_value / 10.0
        self.log_message(f"Saturation updated to: {saturation:.2f}")

    def update_black_point(self, slider_value):
        global black_point
        black_point = slider_value
        self.log_message(f"Black point updated to: {black_point}")

    def update_detection_area_x1(self, val):
        global detection_area
        detection_area[0] = val
        self.log_message(f"Detection area x1 updated to: {detection_area[0]}")

    def update_detection_area_y1(self, val):
        global detection_area
        detection_area[1] = val
        self.log_message(f"Detection area y1 updated to: {detection_area[1]}")

    def update_detection_area_x2(self, val):
        global detection_area
        detection_area[2] = val
        self.log_message(f"Detection area x2 updated to: {detection_area[2]}")

    def update_detection_area_y2(self, val):
        global detection_area
        detection_area[3] = val
        self.log_message(f"Detection area y2 updated to: {detection_area[3]}")

    def reset_video_controls(self):
        global contrast, brightness, saturation, black_point
        contrast = 1.0
        brightness = 0
        saturation = 1.0
        black_point = 0
        self.contrast_slider.setValue(int(contrast * 10))
        self.brightness_slider.setValue(brightness)
        self.saturation_slider.setValue(int(saturation * 10))
        self.black_point_slider.setValue(black_point)
        self.gaussian_blur_slider.setValue(5)
        self.log_message("Video controls reset to default values")

    # PRESETS
    def save_preset(self):
        global threshold, low_threshold, high_threshold, contrast, brightness, saturation
        global black_point, cooldown, cooldown_enabled, filename_prefix, detection_area
        global preset_name, gaussian_blur

        current_preset = self.preset_name_entry.text() or "default"
        preset_dict = {
            "threshold": threshold,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "contrast": contrast,
            "brightness": brightness,
            "saturation": saturation,
            "black_point": black_point,
            "cooldown": cooldown,
            "cooldown_enabled": cooldown_enabled,
            "filename_prefix": filename_prefix,
            "detection_area": detection_area,
            "gaussian_blur": gaussian_blur
        }

        if current_preset in self.available_presets:
            reply = QMessageBox.question(
                self,
                "Confirm Overwrite",
                f"Preset '{current_preset}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        with open(f"{current_preset}.json", "w") as f:
            json.dump(preset_dict, f)

        self.log_message(f"Preset '{current_preset}' saved.")
        if current_preset not in self.available_presets:
            self.available_presets.append(current_preset)
            self.preset_combo.addItem(current_preset)

    def load_preset(self):
        global threshold, low_threshold, high_threshold, contrast, brightness, saturation
        global black_point, cooldown, cooldown_enabled, filename_prefix, detection_area
        global gaussian_blur

        load_name = self.preset_combo.currentText()
        if not load_name:
            self.log_message("No preset selected to load.", "error")
            return

        try:
            with open(f"{load_name}.json", "r") as f:
                loaded = json.load(f)

            threshold = loaded["threshold"]
            low_threshold = loaded["low_threshold"]
            high_threshold = loaded["high_threshold"]
            contrast = loaded["contrast"]
            brightness = loaded["brightness"]
            saturation = loaded["saturation"]
            black_point = loaded["black_point"]
            cooldown = loaded["cooldown"]
            cooldown_enabled = loaded["cooldown_enabled"]
            filename_prefix = loaded["filename_prefix"]
            detection_area = loaded["detection_area"]
            gaussian_blur = loaded.get("gaussian_blur", 5)

            # Update UI
            self.threshold_slider.setValue(threshold)
            self.canny_low_slider.setValue(low_threshold)
            self.canny_high_slider.setValue(high_threshold)
            self.contrast_slider.setValue(int(contrast * 10))
            self.brightness_slider.setValue(brightness)
            self.saturation_slider.setValue(int(saturation * 10))
            self.black_point_slider.setValue(black_point)
            self.cooldown_slider.setValue(cooldown)
            self.filename_entry.setText(filename_prefix)
            self.cooldown_check.setChecked(cooldown_enabled)
            self.detection_area_x1_slider.setValue(detection_area[0])
            self.detection_area_y1_slider.setValue(detection_area[1])
            self.detection_area_x2_slider.setValue(detection_area[2])
            self.detection_area_y2_slider.setValue(detection_area[3])
            self.gaussian_blur_slider.setValue(gaussian_blur)

            self.log_message(f"Preset '{load_name}' loaded.")
        except FileNotFoundError:
            self.log_message(f"Preset file '{load_name}.json' not found.", "error")

    def update_preset_name(self, text):
        global preset_name
        preset_name = text
        self.log_message(f"Preset name updated to: {preset_name}")

    def update_gaussian_blur(self, val):
        global gaussian_blur
        gaussian_blur = val if (val % 2 == 1) else val  # We'll fix oddness at usage time
        self.log_message(f"Gaussian blur updated to: {gaussian_blur}")

    # -------------------------------------------------------------------------
    # MISSING METHODS FOR CANNY MIN/MAX
    # -------------------------------------------------------------------------
    def update_canny_low_min(self, val):
        """
        Sets the minimum possible value for the Canny Low slider.
        E.g., if the user sets it to 20, then the slider range for canny_low_slider is [20..canny_low_max].
        """
        global canny_low_min
        try:
            canny_low_min = int(val)
            self.canny_low_slider.setMinimum(canny_low_min)
            self.log_message(f"Canny low threshold MIN updated to: {canny_low_min}")
        except ValueError:
            pass

    def update_canny_low_max(self, val):
        """
        Sets the maximum possible value for the Canny Low slider.
        """
        global canny_low_max
        try:
            canny_low_max = int(val)
            self.canny_low_slider.setMaximum(canny_low_max)
            self.log_message(f"Canny low threshold MAX updated to: {canny_low_max}")
        except ValueError:
            pass

    def update_canny_high_min(self, val):
        """
        Sets the minimum possible value for the Canny High slider.
        """
        global canny_high_min
        try:
            canny_high_min = int(val)
            self.canny_high_slider.setMinimum(canny_high_min)
            self.log_message(f"Canny high threshold MIN updated to: {canny_high_min}")
        except ValueError:
            pass

    def update_canny_high_max(self, val):
        """
        Sets the maximum possible value for the Canny High slider.
        """
        global canny_high_max
        try:
            canny_high_max = int(val)
            self.canny_high_slider.setMaximum(canny_high_max)
            self.log_message(f"Canny high threshold MAX updated to: {canny_high_max}")
        except ValueError:
            pass

    def closeEvent(self, event):
        global running, cap
        running = False
        if cap and cap.isOpened():
            cap.release()
        self.update_timer.stop()
        self.log_message("Application closed.")
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setWindowIcon(QIcon('ccqticon.png'))


    window = FocusCC()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import time
import json
from tkinter import Tk, Scale, IntVar, Label, Text, Frame, Canvas, StringVar, OptionMenu, Entry, Checkbutton, Button, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import subprocess
import platform
import os

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_CAMERA = 0
THRESHOLD_MIN = 100
THRESHOLD_MAX = 5000
CANNY_LOW_MIN = 10
CANNY_LOW_MAX = 300
CANNY_HIGH_MIN = 50
CANNY_HIGH_MAX = 300

# Global Variables
running = False
save_events = False
only_view = True
threshold = 1000
low_threshold = 50
high_threshold = 150
cap = None
selected_camera = DEFAULT_CAMERA
filename_prefix = "event"
cooldown = 5  # Cooldown in seconds
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
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            device_name = get_camera_name(i)
            available_cameras.append((i, device_name))
            cap.release()
    return available_cameras

def get_camera_name(index):
    """Gets the camera name using system_profiler on macOS or v4l2-ctl on Linux."""
    if platform.system() == "Darwin":  # macOS
        try:
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            camera_name = None
            for i, line in enumerate(lines):
                if 'Model ID' in line:
                    camera_name = lines[i - 1].strip()
                if 'Unique ID' in line and camera_name:
                    return camera_name
        except Exception as e:
            return f"Camera {index}"
    else:  # Linux
        try:
            result = subprocess.run(['v4l2-ctl', '--device', f'/dev/video{index}', '--info'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Card type' in line:
                    return line.split(':')[1].strip()
        except Exception as e:
            return f"Camera {index}"
    return f"Camera {index}"

def update_feed():
    """Continuously updates the video feed in the window."""
    global cap, running, threshold, save_events, only_view, low_threshold, high_threshold, last_event_time, cooldown_enabled, contrast, brightness, saturation, black_point, detection_area, gaussian_blur
    if not cap or not cap.isOpened():
        root.after(100, update_feed)
        return

    ret, frame = cap.read()
    if not ret:
        log_message("Error: Unable to read the video feed.", level="error")
        root.after(100, update_feed)
        return

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Apply video controls
    frame = apply_video_controls(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Apply detection area
    x1, y1, x2, y2 = detection_area
    edges[:y1, :] = 0
    edges[y2:, :] = 0
    edges[:, :x1] = 0
    edges[:, x2:] = 0

    event_detected = np.sum(edges > 0) > threshold
    current_time = time.time()
    if running and event_detected and (not cooldown_enabled or (current_time - last_event_time) > cooldown):
        handle_event(frame)
        last_event_time = current_time

    display_frame(frame, edges)
    root.after(10, update_feed)

def apply_video_controls(frame):
    """Applies video controls to the frame."""
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame = np.clip(frame + black_point, 0, 255).astype(np.uint8)
    return frame

def handle_event(frame):
    """Handles the event detection logic."""
    if save_events:
        filename = f"{filename_prefix}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        log_message(f"Event detected and saved as {filename}")
    if only_view:
        log_message("Event detected, no photo saved.")
    cv2.putText(frame, "EVENT DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def display_frame(frame, edges):
    """Displays the frame and edges in the Tkinter window."""
    x1, y1, x2, y2 = detection_area
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red detection area contour
    combined_view = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)))
    rgb_frame = cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, anchor="nw", image=imgtk)
    video_canvas.imgtk = imgtk

def start_camera():
    """Starts the webcam."""
    global cap, running
    if cap and cap.isOpened():
        log_message("Webcam is already active.")
        return

    cap = cv2.VideoCapture(selected_camera)
    if not cap.isOpened():
        log_message("Error: Unable to access the webcam.", level="error")
        return

    running = True
    log_message(f"Webcam {selected_camera} started.")
    root.after(10, update_feed)  # Start updating the feed

def stop_camera():
    """Stops the video feed."""
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    log_message("Video feed stopped.")

def select_camera(value):
    """Sets the selected camera."""
    global selected_camera, running
    was_running = running
    if running:
        stop_camera()
    selected_camera = int(value.split(' - ')[0])
    log_message(f"Selected camera: {selected_camera}")
    if was_running:
        start_camera()

def toggle_save():
    """Toggles saving of detected events."""
    global save_events, only_view
    save_events = not save_events
    only_view = not save_events
    log_message(f"Save events: {'Enabled' if save_events else 'Disabled'}")

def update_threshold(value):
    """Updates the event threshold from the slider."""
    global threshold
    threshold = int(value)
    log_message(f"Threshold updated to: {threshold}")

def update_canny_low(value):
    """Updates the low Canny threshold."""
    global low_threshold
    low_threshold = int(value)
    log_message(f"Canny low threshold updated to: {low_threshold}")

def update_canny_high(value):
    """Updates the high Canny threshold."""
    global high_threshold
    high_threshold = int(value)
    log_message(f"Canny high threshold updated to: {high_threshold}")

def update_filename_prefix(value):
    """Updates the filename prefix for saved images."""
    global filename_prefix
    filename_prefix = value
    log_message(f"Filename prefix updated to: {filename}")

def update_cooldown(value):
    """Updates the cooldown period between events."""
    global cooldown
    cooldown = int(value)
    log_message(f"Cooldown updated to: {cooldown} seconds")

def toggle_cooldown():
    """Toggles the cooldown feature."""
    global cooldown_enabled
    cooldown_enabled = not cooldown_enabled
    log_message(f"Cooldown: {'Enabled' if cooldown_enabled else 'Disabled'}")

def update_contrast(value):
    """Updates the contrast of the video feed."""
    global contrast
    contrast = float(value)
    log_message(f"Contrast updated to: {contrast}")

def update_brightness(value):
    """Updates the brightness of the video feed."""
    global brightness
    brightness = int(value)
    log_message(f"Brightness updated to: {brightness}")

def update_saturation(value):
    """Updates the saturation of the video feed."""
    global saturation
    saturation = float(value)
    log_message(f"Saturation updated to: {saturation}")

def update_black_point(value):
    """Updates the black point of the video feed."""
    global black_point
    black_point = int(value)
    log_message(f"Black point updated to: {black_point}")

def update_detection_area_x1(value):
    """Updates the x1 coordinate of the detection area."""
    global detection_area
    detection_area[0] = int(value)
    log_message(f"Detection area x1 updated to: {detection_area[0]}")

def update_detection_area_y1(value):
    """Updates the y1 coordinate of the detection area."""
    global detection_area
    detection_area[1] = int(value)
    log_message(f"Detection area y1 updated to: {detection_area[1]}")

def update_detection_area_x2(value):
    """Updates the x2 coordinate of the detection area."""
    global detection_area
    detection_area[2] = int(value)
    log_message(f"Detection area x2 updated to: {detection_area[2]}")

def update_detection_area_y2(value):
    """Updates the y2 coordinate of the detection area."""
    global detection_area
    detection_area[3] = int(value)
    log_message(f"Detection area y2 updated to: {detection_area[3]}")

def reset_video_controls():
    """Resets the video controls to their default values."""
    global contrast, brightness, saturation, black_point
    contrast = 1.0
    brightness = 0
    saturation = 1.0
    black_point = 0
    contrast_slider.set(contrast)
    brightness_slider.set(brightness)
    saturation_slider.set(saturation)
    black_point_slider.set(black_point)
    log_message("Video controls reset to default values")

def save_preset():
    """Saves the current settings to a preset file."""
    preset = {
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
        "detection_area": detection_area
    }
    if preset_name in available_presets:
        if not messagebox.askyesno("Confirm Overwrite", f"Preset '{preset_name}' already exists. Overwrite?"):
            return
    with open(f"{preset_name}.json", "w") as f:
        json.dump(preset, f)
    log_message(f"Preset '{preset_name}' saved")

def load_preset():
    """Loads the settings from a preset file."""
    global threshold, low_threshold, high_threshold, contrast, brightness, saturation, black_point, cooldown, cooldown_enabled, filename_prefix, detection_area
    preset_name = preset_var.get()
    try:
        with open(f"{preset_name}.json", "r") as f:
            preset = json.load(f)
        threshold = preset["threshold"]
        low_threshold = preset["low_threshold"]
        high_threshold = preset["high_threshold"]
        contrast = preset["contrast"]
        brightness = preset["brightness"]
        saturation = preset["saturation"]
        black_point = preset["black_point"]
        cooldown = preset["cooldown"]
        cooldown_enabled = preset["cooldown_enabled"]
        filename_prefix = preset["filename_prefix"]
        detection_area = preset["detection_area"]

        threshold_slider.set(threshold)
        canny_low_slider.set(low_threshold)
        canny_high_slider.set(high_threshold)
        contrast_slider.set(contrast)
        brightness_slider.set(brightness)
        saturation_slider.set(saturation)
        black_point_slider.set(black_point)
        cooldown_slider.set(cooldown)
        filename_entry.delete(0, "end")
        filename_entry.insert(0, filename_prefix)
        cooldown_var.set(1 if cooldown_enabled else 0)
        detection_area_x1_slider.set(detection_area[0])
        detection_area_y1_slider.set(detection_area[1])
        detection_area_x2_slider.set(detection_area[2])
        detection_area_y2_slider.set(detection_area[3])

        log_message(f"Preset '{preset_name}' loaded")
    except FileNotFoundError:
        log_message(f"Preset file '{preset_name}.json' not found", level="error")

def update_preset_name(value):
    """Updates the preset name for saving and loading."""
    global preset_name
    preset_name = value
    log_message(f"Preset name updated to: {preset_name}")

def log_message(message, level="info"):
    """Adds a message to the log window with a specified level."""
    logs.append(f"{level.upper()}: {message}\n")
    log_box.config(state="normal")
    log_box.insert("end", f"{level.upper()}: {message}\n")
    log_box.see("end")
    log_box.config(state="disabled")

def filter_logs(level):
    """Filters the log messages based on the selected level."""
    log_box.config(state="normal")
    log_box.delete(1.0, "end")
    for log in logs:
        if log.startswith(level.upper()):
            log_box.insert("end", log)
    log_box.config(state="disabled")

def exit_program():
    """Exits the program."""
    stop_camera()
    root.quit()

def update_gaussian_blur(value):
    """Updates the Gaussian blur parameter."""
    global gaussian_blur
    gaussian_blur = int(value)
    if gaussian_blur % 2 == 0:
        gaussian_blur += 1  # Ensure the value is odd
    log_message(f"Gaussian blur updated to: {gaussian_blur}")

def update_canny_low_min(value):
    """Updates the minimum value for the low Canny threshold slider."""
    global canny_low_min
    canny_low_min = int(value)
    canny_low_slider.config(from_=canny_low_min)
    log_message(f"Canny low threshold min updated to: {canny_low_min}")

def update_canny_low_max(value):
    """Updates the maximum value for the low Canny threshold slider."""
    global canny_low_max
    canny_low_max = int(value)
    canny_low_slider.config(to=canny_low_max)
    log_message(f"Canny low threshold max updated to: {canny_low_max}")

def update_canny_high_min(value):
    """Updates the minimum value for the high Canny threshold slider."""
    global canny_high_min
    canny_high_min = int(value)
    canny_high_slider.config(from_=canny_high_min)
    log_message(f"Canny high threshold min updated to: {canny_high_min}")

def update_canny_high_max(value):
    """Updates the maximum value for the high Canny threshold slider."""
    global canny_high_max
    canny_high_max = int(value)
    canny_high_slider.config(to=canny_high_max)
    log_message(f"Canny high threshold max updated to: {canny_high_max}")

# GUI
root = Tk()
root.title("FocusCC")

# Set the icon
icon_path = "ccicon.png"  # Update this path to your icon file
icon_image = ImageTk.PhotoImage(file=icon_path)
root.iconphoto(False, icon_image)

# Apply consistent styles
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=5)
style.configure("TCheckbutton", font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12))
style.configure("Red.TButton", font=("Helvetica", 12, "bold"), foreground="black", background="red")

# Title and Subtitle
title_frame = Frame(root, bg="#f0f0f0", bd=2, relief="ridge")
title_frame.pack(fill="x", pady=5)

Label(title_frame, text="FocusCC", font=("Helvetica", 28, "bold"), fg="#4CAF50", bg="#f0f0f0").pack(side="left", padx=10)
Label(title_frame, text="Python-Based Computer Vision for Cloud Chamber Events", font=("Helvetica", 14, "italic"), fg="#555", bg="#f0f0f0").pack(side="left")
Label(title_frame, text="Developed by N. Bagnasco", font=("Helvetica", 12), fg="#555", bg="#f0f0f0").pack(side="left")

exit_button = ttk.Button(title_frame, text="Exit", command=exit_program, style="Red.TButton", compound="left")
exit_button.pack(side="right", padx=10)

# Video feed frame
video_frame = Frame(root, bd=2, relief="sunken", bg="#333")
video_frame.pack(pady=10)

# Left controls container
left_controls_frame = Frame(video_frame, bg="#FFFFE0", bd=2, relief="ridge")
left_controls_frame.pack(side="left", padx=5, pady=5, fill="y")

Label(left_controls_frame, text="Event Threshold", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
threshold_slider = Scale(left_controls_frame, from_=THRESHOLD_MIN, to=THRESHOLD_MAX, orient="horizontal", command=update_threshold, bg="#e0e0e0", highlightthickness=0)
threshold_slider.set(threshold)
threshold_slider.pack()

Label(left_controls_frame, text="Canny Threshold (Low)", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
canny_low_slider = Scale(left_controls_frame, from_=CANNY_LOW_MIN, to=CANNY_LOW_MAX, orient="horizontal", command=update_canny_low, bg="#e0e0e0", highlightthickness=0)
canny_low_slider.set(low_threshold)
canny_low_slider.pack()

Label(left_controls_frame, text="Canny Threshold (High)", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
canny_high_slider = Scale(left_controls_frame, from_=CANNY_HIGH_MIN, to=CANNY_HIGH_MAX, orient="horizontal", command=update_canny_high, bg="#e0e0e0", highlightthickness=0)
canny_high_slider.set(high_threshold)
canny_high_slider.pack()

Label(left_controls_frame, text="Detection Area X1", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
detection_area_x1_slider = Scale(left_controls_frame, from_=0, to=FRAME_WIDTH, orient="horizontal", command=update_detection_area_x1, bg="#e0e0e0", highlightthickness=0)
detection_area_x1_slider.set(detection_area[0])
detection_area_x1_slider.pack()

Label(left_controls_frame, text="Detection Area Y1", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
detection_area_y1_slider = Scale(left_controls_frame, from_=0, to=FRAME_HEIGHT, orient="horizontal", command=update_detection_area_y1, bg="#e0e0e0", highlightthickness=0)
detection_area_y1_slider.set(detection_area[1])
detection_area_y1_slider.pack()

Label(left_controls_frame, text="Detection Area X2", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
detection_area_x2_slider = Scale(left_controls_frame, from_=0, to=FRAME_WIDTH, orient="horizontal", command=update_detection_area_x2, bg="#e0e0e0", highlightthickness=0)
detection_area_x2_slider.set(detection_area[2])
detection_area_x2_slider.pack()

Label(left_controls_frame, text="Detection Area Y2", font=("Helvetica", 12, "bold"), bg="#FFFFE0").pack(pady=5)
detection_area_y2_slider = Scale(left_controls_frame, from_=0, to=FRAME_HEIGHT, orient="horizontal", command=update_detection_area_y2, bg="#e0e0e0", highlightthickness=0)
detection_area_y2_slider.set(detection_area[3])
detection_area_y2_slider.pack()

video_canvas = Canvas(video_frame, width=FRAME_WIDTH * 2, height=FRAME_HEIGHT, bg="#000")
video_canvas.pack(side="left")

# Bottom controls container with scrollbar
bottom_frame_container = Frame(root)
bottom_frame_container.pack(pady=10, fill="both", expand=True)

canvas = Canvas(bottom_frame_container)
scrollbar = ttk.Scrollbar(bottom_frame_container, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# First column: Buttons and Dropdown
button_frame = Frame(scrollable_frame, bg="#f0f0f0", bd=2, relief="ridge")
button_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

ttk.Button(button_frame, text="Start Webcam", command=start_camera, style="TButton").pack(pady=5)
ttk.Button(button_frame, text="Stop Webcam", command=stop_camera, style="TButton").pack(pady=5)

save_var = IntVar(value=0)
ttk.Checkbutton(button_frame, text="Save Events", variable=save_var, command=toggle_save, style="TCheckbutton").pack(pady=5)

# Webcam Selector Dropdown
Label(button_frame, text="Select Camera:", bg="#f0f0f0").pack(pady=5)
camera_var = StringVar(value="0")
available_cameras = list_cameras()
camera_selector = OptionMenu(button_frame, camera_var, *[f"{index} - {name}" for index, name in available_cameras], command=select_camera)
camera_selector.pack(pady=5)

# Second column: Log
log_frame = Frame(scrollable_frame, bd=2, relief="ridge", bg="#000")
log_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

Label(log_frame, text="Event Log", font=("Helvetica", 12, "bold"), bg="#000", fg="#0f0").pack()

log_box = ScrolledText(log_frame, height=12, wrap="word", bg="#000", fg="#0f0", font=("Courier", 10), bd=1, state="disabled")
log_box.pack(fill="both", expand=True, padx=5, pady=5)

log_filter_frame = Frame(log_frame, bg="#000")
log_filter_frame.pack(pady=5)

ttk.Button(log_filter_frame, text="Info", command=lambda: filter_logs("info"), style="TButton").pack(side="left", padx=5)
ttk.Button(log_filter_frame, text="Warning", command=lambda: filter_logs("warning"), style="TButton").pack(side="left", padx=5)
ttk.Button(log_filter_frame, text="Error", command=lambda: filter_logs("error"), style="TButton").pack(side="left", padx=5)

# Third column: Filename prefix and Cooldown
settings_frame = Frame(scrollable_frame, bg="#f0f0f0", bd=2, relief="ridge")
settings_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

Label(settings_frame, text="Filename Prefix", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
filename_entry = Entry(settings_frame, font=("Helvetica", 12))
filename_entry.insert(0, filename_prefix)
filename_entry.pack()
filename_entry.bind("<KeyRelease>", lambda event: update_filename_prefix(filename_entry.get()))

Label(settings_frame, text="Cooldown (seconds)", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
cooldown_slider = Scale(settings_frame, from_=1, to=60, orient="horizontal", command=update_cooldown, bg="#e0e0e0", highlightthickness=0)
cooldown_slider.set(cooldown)
cooldown_slider.pack()

cooldown_var = IntVar(value=1)
Checkbutton(settings_frame, text="Enable Cooldown", variable=cooldown_var, command=toggle_cooldown, bg="#f0f0f0", font=("Helvetica", 12)).pack(pady=5)

# Fourth column: Video controls
video_controls_frame = Frame(scrollable_frame, bg="#f0f0f0", bd=2, relief="ridge")
video_controls_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

Label(video_controls_frame, text="Contrast", font=("Helvetica", 12), bg="#f0f0f0").grid(row=0, column=0, pady=5)
contrast_slider = Scale(video_controls_frame, from_=0.5, to=3.0, resolution=0.1, orient="horizontal", command=update_contrast, bg="#e0e0e0", highlightthickness=0)
contrast_slider.grid(row=1, column=0, pady=5)

Label(video_controls_frame, text="Brightness", font=("Helvetica", 12), bg="#f0f0f0").grid(row=0, column=1, pady=5)
brightness_slider = Scale(video_controls_frame, from_=-100, to=100, orient="horizontal", command=update_brightness, bg="#e0e0e0", highlightthickness=0)
brightness_slider.grid(row=1, column=1, pady=5)

Label(video_controls_frame, text="Saturation", font=("Helvetica", 12), bg="#f0f0f0").grid(row=2, column=0, pady=5)
saturation_slider = Scale(video_controls_frame, from_=0.5, to=3.0, resolution=0.1, orient="horizontal", command=update_saturation, bg="#e0e0e0", highlightthickness=0)
saturation_slider.grid(row=3, column=0, pady=5)

Label(video_controls_frame, text="Black Point", font=("Helvetica", 12), bg="#f0f0f0").grid(row=2, column=1, pady=5)
black_point_slider = Scale(video_controls_frame, from_=-100, to=100, orient="horizontal", command=update_black_point, bg="#e0e0e0", highlightthickness=0)
black_point_slider.grid(row=3, column=1, pady=5)

Button(video_controls_frame, text="Reset Video Controls", command=reset_video_controls, font=("Helvetica", 12)).grid(row=4, column=0, columnspan=2, pady=10)

# Preset controls
preset_frame = Frame(scrollable_frame, bg="#f0f0f0", bd=2, relief="ridge")
preset_frame.grid(row=0, column=4, padx=5, pady=5, sticky="nsew")

Label(preset_frame, text="Preset Name", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
preset_name_entry = Entry(preset_frame, font=("Helvetica", 12))
preset_name_entry.insert(0, preset_name)
preset_name_entry.pack()
preset_name_entry.bind("<KeyRelease>", lambda event: update_preset_name(preset_name_entry.get()))

available_presets = [preset.split(".json")[0] for preset in os.listdir() if preset.endswith(".json")]
preset_var = StringVar(value=available_presets[0] if available_presets else "")
preset_dropdown = OptionMenu(preset_frame, preset_var, *available_presets)
preset_dropdown.pack(pady=5)

ttk.Button(preset_frame, text="Save Preset", command=save_preset, style="TButton").pack(pady=5)
ttk.Button(preset_frame, text="Load Preset", command=load_preset, style="TButton").pack(pady=5)

# Advanced options
advanced_options_frame = Frame(scrollable_frame, bg="#f0f0f0", bd=2, relief="ridge")
advanced_options_frame.grid(row=0, column=5, padx=5, pady=5, sticky="nsew")

Label(advanced_options_frame, text="Gaussian Blur", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
gaussian_blur_slider = Scale(advanced_options_frame, from_=1, to=15, orient="horizontal", command=update_gaussian_blur, bg="#e0e0e0", highlightthickness=0)
gaussian_blur_slider.set(gaussian_blur)
gaussian_blur_slider.pack()

Label(advanced_options_frame, text="Canny Low Min", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_low_min_entry = Entry(advanced_options_frame, font=("Helvetica", 12))
canny_low_min_entry.insert(0, canny_low_min)
canny_low_min_entry.pack()
canny_low_min_entry.bind("<KeyRelease>", lambda event: update_canny_low_min(canny_low_min_entry.get()))

Label(advanced_options_frame, text="Canny Low Max", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_low_max_entry = Entry(advanced_options_frame, font=("Helvetica", 12))
canny_low_max_entry.insert(0, canny_low_max)
canny_low_max_entry.pack()
canny_low_max_entry.bind("<KeyRelease>", lambda event: update_canny_low_max(canny_low_max_entry.get()))

Label(advanced_options_frame, text="Canny High Min", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_high_min_entry = Entry(advanced_options_frame, font=("Helvetica", 12))
canny_high_min_entry.insert(0, canny_high_min)
canny_high_min_entry.pack()
canny_high_min_entry.bind("<KeyRelease>", lambda event: update_canny_high_min(canny_high_min_entry.get()))

Label(advanced_options_frame, text="Canny High Max", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_high_max_entry = Entry(advanced_options_frame, font=("Helvetica", 12))
canny_high_max_entry.insert(0, canny_high_max)
canny_high_max_entry.pack()
canny_high_max_entry.bind("<KeyRelease>", lambda event: update_canny_high_max(canny_high_max_entry.get()))

root.protocol("WM_DELETE_WINDOW", exit_program)
root.mainloop()

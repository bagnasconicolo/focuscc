import cv2
import numpy as np
import time
from tkinter import Tk, Scale, IntVar, Label, Text, Frame, Canvas, StringVar, OptionMenu, Entry, Checkbutton
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess
import platform

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_CAMERA = 0
THRESHOLD_MIN = 100
THRESHOLD_MAX = 5000
CANNY_LOW_MIN = 10
CANNY_LOW_MAX = 100
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
    global cap, running, threshold, save_events, only_view, low_threshold, high_threshold, last_event_time, cooldown_enabled
    if not cap or not cap.isOpened():
        root.after(100, update_feed)
        return

    ret, frame = cap.read()
    if not ret:
        log_message("Error: Unable to read the video feed.")
        root.after(100, update_feed)
        return

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    event_detected = np.sum(edges > 0) > threshold
    current_time = time.time()
    if running and event_detected and (not cooldown_enabled or (current_time - last_event_time) > cooldown):
        handle_event(frame)
        last_event_time = current_time

    display_frame(frame, edges)
    root.after(10, update_feed)

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
        log_message("Error: Unable to access the webcam.")
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
    log_message(f"Filename prefix updated to: {filename_prefix}")

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

def log_message(message):
    """Adds a message to the log window."""
    log_box.config(state="normal")
    log_box.insert("end", f"{message}\n")
    log_box.see("end")
    log_box.config(state="disabled")

def exit_program():
    """Exits the program."""
    stop_camera()
    root.quit()

# GUI
root = Tk()
root.title("CharmCC")

# Set the icon
icon_path = "ccicon.png"  # Update this path to your icon file
icon_image = ImageTk.PhotoImage(file=icon_path)
root.iconphoto(False, icon_image)

# Apply consistent styles
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=5)
style.configure("TCheckbutton", font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12))

# Title and Subtitle
title_frame = Frame(root, bg="#f0f0f0", bd=2, relief="ridge")
title_frame.pack(fill="x", pady=5)

Label(title_frame, text="CharmCC", font=("Helvetica", 28, "bold"), fg="#4CAF50", bg="#f0f0f0").pack(side="left", padx=10)
Label(title_frame, text="Python-Based Computer Vision for Cloud Chamber Events", font=("Helvetica", 14, "italic"), fg="#555", bg="#f0f0f0").pack(side="left")
Label(title_frame, text="Developed by N. Bagnasco", font=("Helvetica", 12), fg="#555", bg="#f0f0f0").pack(side="left")

exit_button = ttk.Button(title_frame, text="Exit", command=exit_program, style="TButton")
exit_button.pack(side="right", padx=10)

# Video feed frame
video_frame = Frame(root, bd=2, relief="sunken", bg="#333")
video_frame.pack(pady=10)
video_canvas = Canvas(video_frame, width=FRAME_WIDTH * 2, height=FRAME_HEIGHT, bg="#000")
video_canvas.pack()

# Bottom controls container
bottom_frame = Frame(root)
bottom_frame.pack(pady=10, fill="both", expand=True)

# First column: Sliders
slider_frame = Frame(bottom_frame, bg="#f0f0f0", bd=2, relief="ridge")
slider_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

Label(slider_frame, text="Event Threshold", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
threshold_slider = Scale(slider_frame, from_=THRESHOLD_MIN, to=THRESHOLD_MAX, orient="horizontal", command=update_threshold, bg="#e0e0e0", highlightthickness=0)
threshold_slider.set(threshold)
threshold_slider.pack()

Label(slider_frame, text="Canny Threshold (Low)", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_low_slider = Scale(slider_frame, from_=CANNY_LOW_MIN, to=CANNY_LOW_MAX, orient="horizontal", command=update_canny_low, bg="#e0e0e0", highlightthickness=0)
canny_low_slider.set(low_threshold)
canny_low_slider.pack()

Label(slider_frame, text="Canny Threshold (High)", font=("Helvetica", 12), bg="#f0f0f0").pack(pady=5)
canny_high_slider = Scale(slider_frame, from_=CANNY_HIGH_MIN, to=CANNY_HIGH_MAX, orient="horizontal", command=update_canny_high, bg="#e0e0e0", highlightthickness=0)
canny_high_slider.set(high_threshold)
canny_high_slider.pack()

# Second column: Buttons and Dropdown
button_frame = Frame(bottom_frame, bg="#f0f0f0", bd=2, relief="ridge")
button_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

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

# Third column: Log
log_frame = Frame(bottom_frame, bd=2, relief="ridge", bg="#000")
log_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

Label(log_frame, text="Event Log", font=("Helvetica", 12, "bold"), bg="#000", fg="#0f0").pack()
log_box = Text(log_frame, height=12, wrap="word", bg="#000", fg="#0f0", font=("Courier", 10), bd=1, state="disabled")
log_box.pack(fill="both", expand=True, padx=5, pady=5)

# Fourth column: Filename prefix and Cooldown
settings_frame = Frame(bottom_frame, bg="#f0f0f0", bd=2, relief="ridge")
settings_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

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

root.protocol("WM_DELETE_WINDOW", exit_program)
root.mainloop()

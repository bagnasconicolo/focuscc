
# CharmCC: Python-Based Computer Vision for Cloud Chamber Events V 1.0.0
**For newer versions check `CHANGELOG.md`**

**Author**: Nicolò Bagnasco  
**Contact**: [nicolo.bagnasco@edu.unito.it](mailto:nicolo.bagnasco@edu.unito.it)

---

## Overview

CharmCC is a Python-based application designed for detecting and analyzing events in a cloud chamber using computer vision techniques. It processes a live video feed, applies edge detection, and identifies particle tracks in real time. The software also supports saving detected events, managing multiple webcams, and customizing various detection parameters.

![image](https://github.com/user-attachments/assets/3c07d29c-4605-4ceb-a0c0-634a581c1f9c)


---

## Features

### Main Features:
- **Live Video Feed**: Displays both raw and processed video feeds in real time.
- **Event Detection**: Detects particle tracks based on edge detection and threshold parameters.
- **Webcam Selection**: Choose from multiple connected webcams via a dropdown menu.
- **Cooldown Feature**: Avoid redundant detections by setting a cooldown time between events.
- **Customizable Parameters**:
  - Event detection threshold
  - Low and high thresholds for Canny edge detection
- **Event Logging**: Logs all system messages and detected events in a console-like interface.
- **Save Events**: Toggle between saving detected events as image files or viewing them live.

---

## System Requirements

- Python 3.8 or newer
- Supported Operating Systems:
  - macOS
  - Linux
  - Windows (partial webcam detection support)
- Required Python libraries:
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `tkinter` (pre-installed with Python)

---

## Installation

### Steps to Install:
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/bagnasconicolo/charmcc]
   cd CharmCC
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python charmcc.py
   ```

---

## Usage

### GUI Overview

1. **Live Feed**:
   - Displays the original and processed video feeds side by side.
   - Detected events are highlighted in the processed feed.

2. **Controls**:
   - **Start Webcam**: Activates the video feed for the selected camera.
   - **Stop Webcam**: Deactivates the video feed.
   - **Save Events**: Toggles saving detected events as `.jpg` files.

3. **Webcam Selector**:
   - Use the dropdown menu to select from available webcams.

4. **Event Log**:
   - Displays system messages and detected events in a console-like interface.

5. **Filename Prefix**:
   - Customize the prefix for saved event files.

6. **Cooldown**:
   - Avoid redundant detections by enabling a cooldown period (in seconds) between events.

---

### Workflow

1. **Start the Webcam**:
   - Select a camera from the dropdown menu.
   - Click **Start Webcam** to begin the live feed.

2. **Adjust Parameters**:
   - Use sliders to customize the detection threshold and Canny edge detection parameters.

3. **Monitor Events**:
   - Detected events will be logged in the Event Log and highlighted in the processed feed.

4. **Save Events**:
   - Enable **Save Events** to save images of detected events.

5. **Stop or Exit**:
   - Use **Stop Webcam** to deactivate the feed or **Exit** to close the application.

---

## Customization

### Parameters:
- **Event Threshold**: Adjusts sensitivity for detecting events.
- **Canny Edge Detection**:
  - **Low Threshold**: Lower bound for detecting edges.
  - **High Threshold**: Upper bound for detecting edges.
- **Filename Prefix**: Set a custom prefix for saved event images.
- **Cooldown**: Set a time delay (in seconds) between consecutive detections.

### File Customization:
- **Icon**: Replace `ccicon.png` with your desired icon file in the project directory.

---

## Troubleshooting

### Common Issues:
1. **Webcam Not Detected**:
   - Ensure the webcam is connected and properly recognized by your operating system.
   - On Linux, ensure `v4l2-ctl` is installed.
   - On macOS, ensure camera permissions are enabled.

2. **No Events Detected**:
   - Lower the **Event Threshold** slider.
   - Adjust **Canny Edge Detection** parameters for better sensitivity.

3. **Application Crash**:
   - Ensure all required libraries are installed.
   - Verify that the correct camera index is selected.

4. **Icon Not Displaying**:
   - Ensure the icon file `ccicon.png` is in the correct directory and properly named.

---

## Future Improvements

- Support for advanced particle track visualization.
- Automated analysis of saved event data.
- Cross-platform webcam detection enhancements for Windows.
- Extended cooldown customization (e.g., per event type).

---

## License

This software is open-source and distributed under the MIT License.

---

## Contact

For support, feedback or contributions, contact **Nicolò Bagnasco**:  
📧 [nicolo.bagnasco@edu.unito.it](mailto:nicolo.bagnasco@edu.unito.it)

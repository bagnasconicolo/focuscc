import sys
import subprocess
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QPalette, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QWidget, QMessageBox
)

class ScriptLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("FocusCC Launcher")
        self.setFixedSize(750, 450)  # Set fixed window size

        # Set background image
        self.set_background_image("launcher.png")  # Replace with your image file

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Spacer to move the icons lower
        main_layout.addStretch(4)

        # Horizontal layout for the boxes
        button_layout = QHBoxLayout()

        # Create each box with an icon and two labels
        button1 = self.create_box_with_icon(
            "ccqticon.png", "Canny FCC", "PyQt Interface", "ccqt.py"
        )
        button2 = self.create_box_with_icon(
            "ccicon.png", "Canny FCC Tk (old)", "Legacy Tk Interface", "cc.py"
        )
        button3 = self.create_box_with_icon(
            "ccmicon.png", "Average Motion FCC", "PyQt Motion Detection", "ccm.py"
        )

        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)

        # Add the horizontal layout to the main layout
        main_layout.addLayout(button_layout)

        # Spacer below the icons
        main_layout.addStretch(1)

    def create_box_with_icon(self, icon_path, label_text1, label_text2, script_name):
        """Creates a bordered box containing an icon and two labels."""
        container = QFrame()
        container.setFrameStyle(QFrame.Box)
        container.setStyleSheet("border: 2px solid black; background-color: white;")
        container.setFixedSize(150, 150)

        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)

        # Icon button
        button = QPushButton()
        button.setIcon(QIcon(icon_path))  # Set the icon
        button.setIconSize(QPixmap(icon_path).size() * 0.4)  # Reduce icon size
        button.setStyleSheet("border: none;")  # Remove button border
        button.clicked.connect(lambda: self.run_script(script_name))
        layout.addWidget(button)

        # First label
        label1 = QLabel(label_text1)
        label1.setAlignment(Qt.AlignCenter)
        label1.setStyleSheet("font-size: 12px; border: none;")  # Clean, simple label
        layout.addWidget(label1)

        # Second label
        label2 = QLabel(label_text2)
        label2.setAlignment(Qt.AlignCenter)
        label2.setStyleSheet("font-size: 12px; color: gray; border: none;")  # Sub-label
        layout.addWidget(label2)

        return container

    def set_background_image(self, image_path):
        """Sets a background image for the window, resized to fit."""
        palette = QPalette()
        pixmap = QPixmap(image_path)
        resized_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        palette.setBrush(QPalette.Window, QBrush(resized_pixmap))
        self.setPalette(palette)

    def run_script(self, script_name):
        """Runs the specified script and closes the window."""
        try:
            subprocess.Popen(["python", script_name])
            self.close()  # Close the window when a script is launched
        except FileNotFoundError:
            self.show_error(f"Error: Script '{script_name}' not found.")
        except Exception as e:
            self.show_error(f"Error while executing the script: {e}")

    def show_error(self, message):
        """Displays an error message."""
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.show()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon('launchericon.png'))

    launcher = ScriptLauncher()
    launcher.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

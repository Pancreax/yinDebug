import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QSlider, QLabel, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class SineWaveGraph:
    def __init__(self, parent=None):
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        
        # Initial frequency
        self.frequency = 1.0
        self.x = np.linspace(0, 2 * np.pi, 500)
        self.line, = self.ax.plot(self.x, np.sin(self.frequency * self.x))
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_title("Sine Wave")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

    def update_frequency(self, frequency):
        # Update frequency and redraw the sine wave
        self.frequency = frequency
        self.line.set_ydata(np.sin(self.frequency * self.x))
        self.canvas.draw()

class FrequencySlider:
    def __init__(self, label_text, callback):
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(1)  # Minimum frequency
        self.slider.setMaximum(100)  # Maximum frequency
        self.slider.setValue(10)  # Initial frequency (10 corresponds to 1.0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(lambda value: callback(value / 10.0))
        self.label = QLabel(f"{label_text}: 1.0 Hz")
        self.callback = callback

    def update_label(self, frequency):
        self.label.setText(f"Frequency: {frequency:.1f} Hz")

class SineWaveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sine Wave Frequency Adjuster")

        # Layout
        layout = QVBoxLayout()

        # Create Graph
        self.graph1 = SineWaveGraph()
        layout.addWidget(self.graph1.canvas)

        self.graph2 = SineWaveGraph()
        layout.addWidget(self.graph2.canvas)

        # Create Slider
        self.slider1 = FrequencySlider("Frequency", self.update_frequency1)
        layout.addWidget(self.slider1.slider)
        layout.addWidget(self.slider1.label)

        # Create Slider
        self.slider2 = FrequencySlider("Frequency", self.update_frequency2)
        layout.addWidget(self.slider2.slider)
        layout.addWidget(self.slider2.label)

        # Set layout
        self.setLayout(layout)

    def update_frequency1(self, frequency):
        self.graph1.update_frequency(frequency)
        self.slider1.update_label(frequency)

    def update_frequency2(self, frequency):
        self.graph2.update_frequency(frequency)
        self.slider2.update_label(frequency)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SineWaveApp()
    window.show()
    sys.exit(app.exec_())

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QSlider, QLabel, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class YinAudioPitchDetector:
    def __init__(self,bufferSize, sampleRate):
        self.sampleRate = sampleRate
        self.bufferSize = bufferSize
        self.mYinBuffer = np.zeros(self.bufferSize)
        self.mDataInputBuffer = np.zeros(self.bufferSize)
        self.mHannHelper = 0.5 - (0.5 * np.cos((2.0 * np.pi * np.arange(bufferSize)) / (bufferSize - 1)))
        self.minFrequency = 20.0

    def process(self, buffer):
        pitch = -1
        self.mYinBuffer = np.zeros(len(self.mYinBuffer))
        
        # Pre-Process: Apply Hann window over the sample
        self.mDataInputBuffer = buffer * self.mHannHelper

        self.mYinBuffer[0] = 1

        runningSum = 0
        delta = 0
        tauEstimate = -1
        self.maxFindRange = int(self.sampleRate / self.minFrequency)

        for tau in range(1, self.maxFindRange):
            # Yin Algorithm: Difference function
            for j in range(len(self.mYinBuffer)):
                delta = self.mDataInputBuffer[j] - self.mDataInputBuffer[(j + tau)%len(self.mYinBuffer)]
                self.mYinBuffer[tau] += delta * delta

            # Yin Algorithm: Cumulative mean normalized difference function
            runningSum += self.mYinBuffer[tau]
            self.mYinBuffer[tau] *= tau / runningSum

            # Yin Algorithm: Absolute threshold
            #if mThresholdOk:
            #    if self.mYinBuffer[tau - 1] <= self.mYinBuffer[tau]:
            #        tauEstimate = tau - 1
            #        break
            #elif self.mYinBuffer[tau] < self.currentThreshold:
            #    mThresholdOk = True



class SinBuffer:
    def __init__(self, bufferSize, sampleRate, freqs=[1,2]):
        self.bufferSize = bufferSize
        self.sampleRate = 8000
        self.x = np.linspace(0, bufferSize -1, bufferSize)
        self.y = self.generate_sinusoids(freqs)

    def generate_sinusoids(self, freqs):
        y = np.zeros(self.bufferSize)
        for freq in freqs:
            y += np.sin(2 * np.pi * freq * self.x / self.sampleRate)
        return y / len(freqs)

class GenericBuffer:
    def __init__(self, bufferSize):
        self.x = np.linspace(0, bufferSize -1, bufferSize)
        self.y = np.zeros(bufferSize)

class SineWaveGraph:
    def __init__(self, bufferSize, sampleRate=8000, parent=None):
        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        initialBuffer = SinBuffer(bufferSize, sampleRate)
        self.line, = self.ax.plot(initialBuffer.x, initialBuffer.y)
        
        # Initial frequency
        self.update_buffer(initialBuffer.x, initialBuffer.y)

    def update_buffer(self, x, y):
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_title("Sine Wave")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

class FrequencySlider:
    def __init__(self, label_text, callback):
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(1)  # Minimum frequency
        self.slider.setMaximum(1000)  # Maximum frequency
        self.slider.setValue(1)  # Initial frequency (10 corresponds to 1.0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(lambda value: callback(value))
        self.label = QLabel(f"{label_text}: 1.0 Hz")
        self.callback = callback

    def update_label(self, frequency):
        self.label.setText(f"Frequency: {frequency:.1f} Hz")

class SineWaveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sine Wave Frequency Adjuster")

        self.bufferSize = 1000
        self.sampleRate = 8000

        # Layout
        layout = QVBoxLayout()

        # Create Graph
        self.graph0 = SineWaveGraph(bufferSize=self.bufferSize)
        layout.addWidget(self.graph0.canvas)

        self.graph1 = SineWaveGraph(bufferSize=self.bufferSize)
        layout.addWidget(self.graph1.canvas)

        self.graph2 = SineWaveGraph(bufferSize=400)
        layout.addWidget(self.graph2.canvas)

        # Create Slider
        self.slider0 = FrequencySlider("Frequency", self.update_frequency0)
        layout.addWidget(self.slider0.slider)
        layout.addWidget(self.slider0.label)

        # Create Slider
        self.slider1 = FrequencySlider("Frequency", self.update_frequency1)
        layout.addWidget(self.slider1.slider)
        layout.addWidget(self.slider1.label)

        self.pitchLabel = QLabel("Hz")
        layout.addWidget(self.pitchLabel)

        # Set layout
        self.setLayout(layout)

        self.freqs = [1,2]

        self.yin = YinAudioPitchDetector(bufferSize=self.bufferSize, sampleRate=self.sampleRate)

        self.update()

    def update_frequency0(self, frequency):
        self.freqs[0] = frequency
        self.update()

    def update_frequency1(self, frequency):
        self.freqs[1] = frequency
        self.update()

    def update(self):
        buffer = SinBuffer(bufferSize=self.bufferSize, sampleRate=self.sampleRate, freqs=self.freqs)
        self.yin.process(buffer.y)
        self.graph0.update_buffer(buffer.x, buffer.y)
        self.graph1.update_buffer(buffer.x, self.yin.mDataInputBuffer)
        self.graph2.update_buffer(self.sampleRate/buffer.x, self.yin.mYinBuffer)
        self.slider0.update_label(self.freqs[0])
        self.slider1.update_label(self.freqs[1])

        min_index = np.argmin(self.yin.mYinBuffer[:self.yin.maxFindRange])
        frequency = self.sampleRate/min_index
        self.pitchLabel.setText(f"Frequency: {frequency:.2f} Hz")

        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SineWaveApp()
    window.show()
    sys.exit(app.exec_())

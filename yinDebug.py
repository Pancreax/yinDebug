import threading
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QWidget
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
        self.currentThreshold = 0.12

    def process(self, buffer):
        pitch = -1
        self.mYinBuffer = np.zeros(len(self.mYinBuffer))
        
        # Pre-Process: Apply Hann window over the sample
        self.mDataInputBuffer = buffer * self.mHannHelper

        self.mYinBuffer[0] = 1

        runningSum = 0
        delta = 0
        self.tauEstimate = -1
        self.maxFindRange = int(self.sampleRate / self.minFrequency)
        mThresholdOk = False

        for tau in range(1, self.maxFindRange):
            # Yin Algorithm: Difference function
            for j in range(len(self.mYinBuffer)):
                delta = self.mDataInputBuffer[j] - self.mDataInputBuffer[(j + tau)%len(self.mYinBuffer)]
                self.mYinBuffer[tau] += delta * delta

            # Yin Algorithm: Cumulative mean normalized difference function
            runningSum += self.mYinBuffer[tau]
            self.mYinBuffer[tau] *= tau / runningSum

        for tau in range(1, self.maxFindRange):
            # Yin Algorithm: Absolute threshold
            if mThresholdOk:
                if self.mYinBuffer[tau - 1] <= self.mYinBuffer[tau]:
                    self.tauEstimate = tau - 1
                    break
            elif self.mYinBuffer[tau] < self.currentThreshold:
                mThresholdOk = True

        self.betterTau = -1
        if self.tauEstimate != -1:
            x0 = self.tauEstimate if self.tauEstimate < 1 else self.tauEstimate - 1
            x2 = self.tauEstimate + 1 if self.tauEstimate + 1 < len(self.mYinBuffer) else self.tauEstimate

            if x0 == self.tauEstimate:
                self.betterTau = self.tauEstimate if self.mYinBuffer[self.tauEstimate] <= self.mYinBuffer[x2] else x2
            elif x2 == self.tauEstimate:
                self.betterTau = self.tauEstimate if self.mYinBuffer[self.tauEstimate] <= self.mYinBuffer[x0] else x0
            else:
                s0 = self.mYinBuffer[x0]
                s1 = self.mYinBuffer[self.tauEstimate]
                s2 = self.mYinBuffer[x2]

                self.betterTau = self.tauEstimate + 0.5 * (s2 - s0) / (2.0 * s1 - s2 - s0)



class SinBuffer:
    def __init__(self, bufferSize, sampleRate, freqs=[1,2], amps=[1.0,1.0]):
        self.bufferSize = bufferSize
        self.sampleRate = 8000
        self.x = np.linspace(0, bufferSize -1, bufferSize)
        self.y = self.generate_sinusoids(freqs, amps)

    def generate_sinusoids(self, freqs, amps):
        y = np.zeros(self.bufferSize)
        for i in range(len(freqs)):
            y += np.sin(2 * np.pi * freqs[i] * self.x / self.sampleRate) * amps[i]
        return y / len(freqs)

class GenericBuffer:
    def __init__(self, bufferSize):
        self.x = np.linspace(0, bufferSize -1, bufferSize)
        self.y = np.zeros(bufferSize)

class SineWaveGraph:
    def __init__(self, title, bufferSize, ymin=-1.5, ymax=1.5, sampleRate=8000, parent=None):
        # Matplotlib Figure
        self.ymax = ymax
        self.ymin = ymin
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        initialBuffer = SinBuffer(bufferSize, sampleRate)
        self.line, = self.ax.plot(initialBuffer.x, initialBuffer.y)
        
        # Initial frequency
        self.update_buffer(initialBuffer.x, initialBuffer.y)

    def update_buffer(self, x, y):
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_title(self.title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

class FrequencySlider:
    def __init__(self, label_text, callbackF, callbackA):

        self.layout = QVBoxLayout()
        self.fslider = QSlider()
        self.fslider.setOrientation(Qt.Horizontal)
        self.fslider.setMinimum(1)  # Minimum frequency
        self.fslider.setMaximum(1000)  # Maximum frequency
        self.fslider.setValue(1)  
        self.fslider.setTickInterval(1)
        self.fslider.valueChanged.connect(lambda value: callbackF(value))

        self.aslider = QSlider()
        self.aslider.setOrientation(Qt.Horizontal)
        self.aslider.setMinimum(0)  # Minimum amplitude
        self.aslider.setMaximum(1000)  # Maximum amplitude
        self.aslider.setValue(1000)  
        self.aslider.setTickInterval(1)
        self.aslider.valueChanged.connect(lambda value: callbackA(value/1000))

        self.label = QLabel(f"{label_text}: 1.0 Hz")

        self.layout.addWidget(self.fslider)
        self.layout.addWidget(self.aslider)
        self.layout.addWidget(self.label)

    def update_label(self, frequency, amplitude):
        self.label.setText(f"{frequency:.1f}Hz {amplitude:.2} V")

class SineWaveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sine Wave Frequency Adjuster")

        self.bufferSize = 1000
        self.sampleRate = 8000

        # Layout
        main_layout = QVBoxLayout()

        v_layout0 = QVBoxLayout()
        v_layout1 = QVBoxLayout()
        h_layout = QHBoxLayout()

        # Create Graph
        self.graph0 = SineWaveGraph(title="input", bufferSize=self.bufferSize)
        v_layout0.addWidget(self.graph0.canvas)

        self.graph1 = SineWaveGraph(title="input windowed", bufferSize=self.bufferSize)
        v_layout0.addWidget(self.graph1.canvas)

        self.graph2 = SineWaveGraph(title="output tau",bufferSize=500, ymin=0, ymax=3)
        v_layout1.addWidget(self.graph2.canvas)

        self.graph3 = SineWaveGraph(title="output Hz",bufferSize=1500, ymin=0, ymax=3)
        v_layout1.addWidget(self.graph3.canvas)

        h_layout.addLayout(v_layout0),
        h_layout.addLayout(v_layout1),
        main_layout.addLayout(h_layout)

        sliders_h_layout = QHBoxLayout()

        # Create Slider
        self.slider0 = FrequencySlider("Frequency", self.update_frequency0, self.update_amplitude0)
        sliders_h_layout.addLayout(self.slider0.layout)

        # Create Slider
        self.slider1 = FrequencySlider("Frequency", self.update_frequency1, self.update_amplitude1)
        sliders_h_layout.addLayout(self.slider1.layout)

        self.pitchLabelByMin = QLabel("Hz")
        self.tauLabelByMin = QLabel("Tau")
        self.pitchLabelByThreshold = QLabel("Hz")
        self.tauLabelByThreshold = QLabel("Tau")
        self.pitchLabelByBetterTau = QLabel("Hz")
        self.tauLabelByBetterTau = QLabel("Tau")

        main_layout.addLayout(sliders_h_layout)

        main_layout.addWidget(self.pitchLabelByMin)
        main_layout.addWidget(self.tauLabelByMin)
        main_layout.addWidget(self.pitchLabelByThreshold)
        main_layout.addWidget(self.tauLabelByThreshold)
        main_layout.addWidget(self.pitchLabelByBetterTau)
        main_layout.addWidget(self.tauLabelByBetterTau)
        # Set layout
        self.setLayout(main_layout)

        self.freqs = [1,2]
        self.amps = [1.0,1.0]

        self.yin = YinAudioPitchDetector(bufferSize=self.bufferSize, sampleRate=self.sampleRate)

        self.update()

    def update_frequency0(self, frequency):
        self.freqs[0] = frequency
        self.update()
    
    def update_amplitude0(self, amplitude):
        self.amps[0] = amplitude
        self.update()

    def update_frequency1(self, frequency):
        self.freqs[1] = frequency
        self.update()

    def update_amplitude1(self, amplitude):
        self.amps[1] = amplitude
        self.update()

    def update(self):
        buffer = SinBuffer(bufferSize=self.bufferSize, sampleRate=self.sampleRate, freqs=self.freqs, amps=self.amps)
        self.yin.process(buffer.y)
        self.graph0.update_buffer(buffer.x, buffer.y)
        self.graph1.update_buffer(buffer.x, self.yin.mDataInputBuffer)
        self.graph2.update_buffer(buffer.x, self.yin.mYinBuffer)
        self.graph3.update_buffer(self.sampleRate/buffer.x, self.yin.mYinBuffer)
        self.slider0.update_label(self.freqs[0], self.amps[0])
        self.slider1.update_label(self.freqs[1], self.amps[1])

        min_index = np.argmin(self.yin.mYinBuffer[:self.yin.maxFindRange])
        frequency = self.sampleRate/min_index
        self.pitchLabelByMin.setText(f"Frequency(by min): {frequency:.2f} Hz")
        self.tauLabelByMin.setText(f"Tau: (by min) {min_index:.2f}")
        self.pitchLabelByThreshold.setText(f"Frequency(by threshold): {self.sampleRate/self.yin.tauEstimate:.2f} Hz")
        self.tauLabelByThreshold.setText(f"Tau: (by threshold) {self.yin.tauEstimate:.2f}")
        self.pitchLabelByBetterTau.setText(f"Frequency(by better tau): {self.sampleRate/self.yin.betterTau:.2f} Hz")
        self.tauLabelByBetterTau.setText(f"Tau: (by better tau) {self.yin.betterTau:.2f}")

def listen_for_exit(app):
    while True:
        command = input("Type 'q' to quit: ")
        if command.strip().lower() == 'q':
            print("Exiting...")
            app.quit()
            break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SineWaveApp()
    window.show()

    # Start a thread to listen for 'q' input
    thread = threading.Thread(target=listen_for_exit, args=(app,), daemon=True)
    thread.start()

    sys.exit(app.exec_())

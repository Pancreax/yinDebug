import threading
import sys
import termios
import tty
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QWidget, QComboBox, QGridLayout, QBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class YinAudioPitchDetector:
    def __init__(self,bufferSize, sampleRate,windowFactor,windowSize):
        self.windowFactor = windowFactor
        self.sampleRate = sampleRate
        self.bufferSize = int(bufferSize)
        self.mYinBuffer = np.zeros(self.bufferSize)
        self.mDataInputBuffer = np.zeros(self.bufferSize)
        self.mHannHelper = 0.5 - (0.5 * np.cos((2.0 * np.pi * np.arange(bufferSize)) / (bufferSize*windowSize - 1)))
        self.minFrequency = 20.0
        self.currentThreshold = 0.12

    def process(self, buffer):
        pitch = -1
        self.mYinBuffer = np.zeros(len(self.mYinBuffer))
        
        # Pre-Process: Apply Hann window over the sample
        self.mDataInputBuffer = buffer * self.mHannHelper * self.windowFactor + buffer * (1 - self.windowFactor)

        self.mYinBuffer[0] = 1

        runningSum = 0
        delta = 0
        self.tauEstimate = -1
        self.maxFindRange = min(int(self.sampleRate / self.minFrequency), len(self.mYinBuffer))
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

class SampleRateDropdown:
    def __init__(self, label_text, callback, options = ["8000", "11025", "16000", "22050", "44100"], parentLayout:QBoxLayout=None):
        self.dropdown = QComboBox()
        self.dropdown.addItems(options)
        self.dropdown.currentTextChanged.connect(callback)
        self.label = QLabel(label_text)
        if parentLayout is not None:
            parentLayout.addWidget(self.label)
            parentLayout.addWidget(self.dropdown)

class SinBuffer:
    def __init__(self, bufferSize, sampleRate, freqs=[1,2], amps=[1.0,1.0]):
        self.bufferSize = bufferSize
        self.sampleRate = sampleRate
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

    def update_buffer(self, x, y, minX=None, maxX=None):
        self.line.set_xdata(x)
        self.line.set_ydata(y)

        self.ax.set_xlim(minX or min(x), maxX or max(x))
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_title(self.title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

class BorderedWidget():
    def __init__(self, parentLayout:QBoxLayout=None):

        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        self.widget.setObjectName("panzerina")

        # Apply a border via stylesheet
        self.widget.setStyleSheet("""
            QWidget#panzerina {
                border: 1px solid #5A5A5A;
                border-radius: 10px;
            }
        """)

        if parentLayout is not None:
            parentLayout.addWidget(self.widget)
    
    def addWidget(self, child:QWidget):
        self.layout.addWidget(child)

class FactorSlider:
    def __init__(self, factor, callback, multiplier=1.0, min=0.0, max=1000.0, parentLayout:QBoxLayout=None):
        self.layout = QVBoxLayout()

        self.label = QLabel("")
        self.layout.addWidget(self.label)

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(int(min))  
        self.slider.setMaximum(int(max))  
        self.slider.setValue(int(factor*1000))  
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(lambda value: callback(value/1000))
        self.layout.addWidget(self.slider)

        if parentLayout is not None:
            parentLayout.addLayout(self.layout)


    def update_label(self, text: str):
        self.label.setText(text)

class FrequencySlider:
    def __init__(self, label_text, freq, amp, callbackF, callbackA, parentLayout:QBoxLayout=None):

        self.layout = QVBoxLayout()
        #self.layout.setAlignment(Qt.AlignTop)
        self.fslider = QSlider()
        self.fslider.setOrientation(Qt.Horizontal)
        self.fslider.setMinimum(1)  # Minimum frequency
        self.fslider.setMaximum(1000)  # Maximum frequency
        self.fslider.setValue(freq)  
        self.fslider.setTickInterval(1)
        self.fslider.valueChanged.connect(lambda value: callbackF(value))

        self.aslider = QSlider()
        self.aslider.setOrientation(Qt.Horizontal)
        self.aslider.setMinimum(0)  # Minimum amplitude
        self.aslider.setMaximum(1000)  # Maximum amplitude
        self.aslider.setValue(int(amp*1000))  
        self.aslider.setTickInterval(1)
        self.aslider.valueChanged.connect(lambda value: callbackA(value/1000))

        self.label = QLabel(f"{label_text}: 1.0 Hz")

        self.layout.addWidget(self.fslider)
        self.layout.addWidget(self.aslider)
        self.layout.addWidget(self.label)

        if parentLayout is not None:
            parentLayout.addLayout(self.layout)

    def update_label(self, frequency, tau, amplitude):
        self.label.setText(f"{frequency:.1f}Hz {tau:.1f}tau {amplitude:.2} V")

class SineWaveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sine Wave Frequency Adjuster")

        self.windowSize = 1.0
        self.windowFactor = 1.0
        self.bufferSize = 1000
        self.sampleRate = 8000
        self.bufferSizeFactor = self.bufferSize/self.sampleRate
        self.freqs = [82,557]
        self.amps = [1.0,0.31]

        # Layout
        output_layout = QVBoxLayout()
        main_layout = QHBoxLayout()
        input_layout = QVBoxLayout()

        input_layout.setAlignment(Qt.AlignTop)

        bufferSizeSampleRateBox = BorderedWidget(input_layout)

        # Create Dropdown Menu for Sample Rates
        self.sample_rate_dropdown = SampleRateDropdown("Sample Rate", self.change_sample_rate, parentLayout=bufferSizeSampleRateBox.layout)

        self.bufferSizeSlider = FactorSlider(self.bufferSizeFactor, 
                                             multiplier=self.sampleRate, 
                                             callback=self.updateBufferSize, 
                                             min=1,
                                             max=500,
                                             parentLayout=bufferSizeSampleRateBox.layout)

        self.windowSlider = FactorSlider(self.windowFactor, 
                                         callback=self.updateWindowFactor, 
                                         parentLayout=bufferSizeSampleRateBox.layout)

        self.windowSizeSlider = FactorSlider(self.windowSize, 
                                             min=1000,
                                             max=2000,
                                             callback=self.updateWindowSize, 
                                             parentLayout=bufferSizeSampleRateBox.layout)

        sliders_h_layout = QVBoxLayout()
        sliders_h_layout.setAlignment(Qt.AlignTop)

        # Create Slider
        self.slider0 = FrequencySlider("Frequency", 
                                       self.freqs[0], 
                                       self.amps[0], 
                                       self.update_frequency0, 
                                       self.update_amplitude0, 
                                       BorderedWidget(sliders_h_layout).layout
                                       )

        # Create Slider
        self.slider1 = FrequencySlider("Frequency", 
                                       self.freqs[1], 
                                       self.amps[1], 
                                       self.update_frequency1, 
                                       self.update_amplitude1, 
                                       BorderedWidget(sliders_h_layout).layout
                                       )

        input_layout.addLayout(sliders_h_layout)

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
        output_layout.addLayout(h_layout)
        self.pitchLabelByMin = QLabel("Hz")
        self.tauLabelByMin = QLabel("Tau")
        self.pitchLabelByThreshold = QLabel("Hz")
        self.tauLabelByThreshold = QLabel("Tau")
        self.pitchLabelByBetterTau = QLabel("Hz")
        self.tauLabelByBetterTau = QLabel("Tau")

        resultLayouts = [QVBoxLayout(), QVBoxLayout(),QVBoxLayout()]

        resultLayouts[0].addWidget(self.pitchLabelByMin)
        resultLayouts[0].addWidget(self.tauLabelByMin)
        resultLayouts[1].addWidget(self.pitchLabelByThreshold)
        resultLayouts[1].addWidget(self.tauLabelByThreshold)
        resultLayouts[2].addWidget(self.pitchLabelByBetterTau)
        resultLayouts[2].addWidget(self.tauLabelByBetterTau)

        resultGlayout = QGridLayout()

        resultGlayout.addLayout(resultLayouts[0], 0, 0)
        resultGlayout.addLayout(resultLayouts[1], 0, 1)
        resultGlayout.addLayout(resultLayouts[2], 0, 2)


        output_layout.addLayout(resultGlayout)

        #main_layout.addWidget(input_layout_frame)
        main_layout.addLayout(input_layout,1)
        main_layout.addLayout(output_layout,3)
        # Set layout
        self.setLayout(main_layout)
        self.update()

    def updateWindowSize(self, windowSize):
        self.windowSize = windowSize
        self.update()

    def updateWindowFactor(self, windowFactor):
        self.windowFactor = windowFactor
        self.update()

    def updateBufferSize(self, bufferSizeFactor):
        self.bufferSizeFactor = bufferSizeFactor
        self.update()

    def change_sample_rate(self, selected_rate):
        #print(f"Sample rate changed to: {selected_rate}")
        # Implement additional logic for handling the selected sample rate if needed
        self.sampleRate = float(selected_rate)
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
        self.bufferSize = int(self.sampleRate * self.bufferSizeFactor)
        self.yin = YinAudioPitchDetector(bufferSize=self.bufferSize, 
                                         sampleRate=self.sampleRate, 
                                         windowFactor=self.windowFactor, 
                                         windowSize=self.windowSize
                                         )
        buffer = SinBuffer(bufferSize=self.bufferSize, sampleRate=self.sampleRate, freqs=self.freqs, amps=self.amps)
        self.yin.process(buffer.y)
        self.graph0.update_buffer(buffer.x, buffer.y)
        self.graph1.update_buffer(buffer.x, self.yin.mDataInputBuffer)
        self.graph2.update_buffer(buffer.x, self.yin.mYinBuffer, maxX=self.yin.maxFindRange)
        self.graph3.update_buffer(self.sampleRate/buffer.x[1:], self.yin.mYinBuffer[1:], minX=20 ,maxX=1000)
        self.slider0.update_label(self.freqs[0], self.sampleRate/self.freqs[0], self.amps[0])
        self.slider1.update_label(self.freqs[1], self.sampleRate/self.freqs[1], self.amps[1])
        self.bufferSizeSlider.update_label(f"BufferSize: {self.bufferSize} {self.bufferSizeFactor*1000}ms")
        self.windowSlider.update_label(f"Window Factor: {self.windowFactor}")
        self.windowSizeSlider.update_label(f"Window size: {self.windowSize}")

        min_index = max(np.argmin(self.yin.mYinBuffer[:self.yin.maxFindRange]),1)
        frequency = self.sampleRate/min_index
        self.pitchLabelByMin.setText(f"Frequency(by min): {frequency:.2f} Hz")
        self.tauLabelByMin.setText(f"Tau: (by min) {min_index:.2f}")
        self.pitchLabelByThreshold.setText(f"Frequency(by threshold): {self.sampleRate/self.yin.tauEstimate:.2f} Hz")
        self.tauLabelByThreshold.setText(f"Tau: (by threshold) {self.yin.tauEstimate:.2f}")
        self.pitchLabelByBetterTau.setText(f"Frequency(by better tau): {self.sampleRate/self.yin.betterTau:.2f} Hz")
        self.tauLabelByBetterTau.setText(f"Tau: (by better tau) {self.yin.betterTau:.2f}")

def listen_for_quit():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'l':
                os._exit(0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SineWaveApp()
    window.show()

    # Start a thread to listen for 'q' key press
    quit_thread = threading.Thread(target=listen_for_quit, daemon=True)
    quit_thread.start()

    sys.exit(app.exec_())

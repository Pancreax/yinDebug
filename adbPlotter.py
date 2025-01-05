import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import re
import subprocess
import threading

from matplotlib.widgets import Slider

class DualBufferGraph:
    def __init__(self, input_title, output_title):
        # Input graph
        self.fig, (self.ax_input, self.ax_output) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.subplots_adjust(left=0.1, bottom=0.3, top=0.95)  # Adjust for sliders

        # Initial parameters
        self.ymin_input = -100
        self.ymax_input = 100
        self.samples_input = 600

        self.ymin_output = 0.0
        self.ymax_output = 2.0
        self.samples_output = 400

        # Plot initial data
        self.input_line, = self.ax_input.plot(range(self.samples_input), [0] * self.samples_input, label="Input")
        self.output_line, = self.ax_output.plot(range(self.samples_output), [0] * self.samples_output, label="Output")

        self.ax_input.set_title(input_title)
        self.ax_output.set_title(output_title)

        self.ax_input.set_xlim(0, self.samples_input)
        self.ax_input.set_ylim(self.ymin_input, self.ymax_input)

        self.ax_output.set_xlim(0, self.samples_output)
        self.ax_output.set_ylim(self.ymin_output, self.ymax_output)

        # Sliders for Input Y Scale
        self.ax_ymin_input = plt.axes([0.1, 0.2, 0.35, 0.03], facecolor="lightgoldenrodyellow")
        self.slider_ymin_input = Slider(self.ax_ymin_input, "Input Y ±", 10, 33000, valinit=self.ymax_input)
        self.slider_ymin_input.on_changed(self.update_input_y_scale)

        # Slider for Input Samples
        self.ax_samples_input = plt.axes([0.55, 0.2, 0.35, 0.03], facecolor="lightgoldenrodyellow")
        self.slider_samples_input = Slider(self.ax_samples_input, "Input Samples", 10, 900, valinit=self.samples_input)
        self.slider_samples_input.on_changed(self.update_input_samples)

        # Sliders for Output Y Scale
        self.ax_ymin_output = plt.axes([0.1, 0.1, 0.35, 0.03], facecolor="lightblue")
        self.slider_ymin_output = Slider(self.ax_ymin_output, "Output Y ±", 0.1, 2.0, valinit=self.ymax_output)
        self.slider_ymin_output.on_changed(self.update_output_y_scale)

        # Slider for Output Samples
        self.ax_samples_output = plt.axes([0.55, 0.1, 0.35, 0.03], facecolor="lightblue")
        self.slider_samples_output = Slider(self.ax_samples_output, "Output Samples", 10, 900, valinit=self.samples_output)
        self.slider_samples_output.on_changed(self.update_output_samples)

    def update_input_y_scale(self, val):
        """Update the Y scale for the input graph symmetrically."""
        scale = self.slider_ymin_input.val
        self.ymin_input = -scale
        self.ymax_input = scale
        self.ax_input.set_ylim(self.ymin_input, self.ymax_input)
        self.fig.canvas.draw_idle()

    def update_input_samples(self, val):
        """Update the number of samples displayed for the input graph."""
        self.samples_input = int(self.slider_samples_input.val)
        #self.input_line.set_xdata(range(self.samples_input))
        #self.ax_input.set_xlim(0, self.samples_input)
        #self.fig.canvas.draw_idle()

    def update_output_y_scale(self, val):
        """Update the Y scale for the output graph symmetrically."""
        scale = self.slider_ymin_output.val
        self.ymin_output = 0
        self.ymax_output = scale
        self.ax_output.set_ylim(self.ymin_output, self.ymax_output)
        self.fig.canvas.draw_idle()

    def update_output_samples(self, val):
        """Update the number of samples displayed for the output graph."""
        self.samples_output = int(self.slider_samples_output.val)
        #self.output_line.set_xdata(range(self.samples_output))
        #self.ax_output.set_xlim(0, self.samples_output)
        #self.fig.canvas.draw_idle()

    def update_buffers(self, input_buffer, output_buffer):
        """Update the graphs with new buffers."""
        inputLenght = min(len(input_buffer) ,self.samples_input)
        outputLenght = min(len(output_buffer) ,self.samples_output)


        self.ax_input.set_xlim(0, inputLenght)
        self.ax_output.set_xlim(0, outputLenght)

        self.input_line.set_ydata(input_buffer[:inputLenght])
        self.input_line.set_xdata(range(inputLenght))

        self.output_line.set_ydata(output_buffer[:outputLenght])
        self.output_line.set_xdata(range(outputLenght))

        self.fig.canvas.draw_idle()


    def show(self):
        plt.show()

def is_valid_number(s):
    """Check if a string is a valid number."""
    s = s.strip()  # Remove any leading/trailing spaces
    if not s:  # Skip empty strings
        return False
    try:
        float(s)  # Try converting to float
        return True
    except ValueError:
        return False


def parse_logcat_line(line):
    """Parse a single logcat line for either input or output buffer."""
    input_match = re.search(r"input: \[(.*?)\]?$", line)
    output_match = re.search(r"output: \[(.*?)\]?$", line)

    if input_match:
        # Parse floating-point numbers for the input buffer
        input_buffer = list(map(float, filter(is_valid_number, input_match.group(1).split(", "))))
        return "input", input_buffer

    if output_match:
        # Parse floating-point numbers for the output buffer
        output_buffer = list(map(float, filter(is_valid_number, output_match.group(1).split(", "))))
        return "output", output_buffer

    return None, None


def read_logcat(graph):
    """Read logcat output in real-time and match input to output buffers."""
    process = subprocess.Popen(
        ["adb", "logcat"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffering
    )

    last_input_buffer = None
    last_output_buffer = None

    for line in iter(process.stdout.readline, ''):
        if "👹 input" in line or "👹 output" in line:  # Filter only relevant lines
            print(f"DEBUG: {line.strip()}")  # Log the relevant line
            buffer_type, buffer_data = parse_logcat_line(line)

            if buffer_type == "input":
                last_input_buffer = buffer_data
                print(f"Parsed Input: {last_input_buffer[:5]}... ({len(last_input_buffer)} samples)")

            elif buffer_type == "output":
                print(f"Parsed Output: {buffer_data[:5]}... ({len(buffer_data)} samples)")
                last_output_buffer = buffer_data

            if last_input_buffer is not None and last_output_buffer is not None:
                graph.update_buffers(last_input_buffer, last_output_buffer)



if __name__ == "__main__":
    BUFFER_SIZE = 800  # Adjust this to the expected buffer size
    graph = DualBufferGraph("Input", "Output")

    # Start a thread to read logcat output
    logcat_thread = threading.Thread(target=read_logcat, args=(graph,), daemon=True)
    logcat_thread.start()

    # Show the plots
    graph.show()

# Dynamic Data Visualization Tool

Welcome to the Dynamic Data Visualization Tool! This Streamlit application allows you to interactively visualize various types of data and mathematical functions. Whether you're interested in plotting mathematical functions, generating waveforms, uploading and analyzing data, solving equations, or visualizing physical phenomena such as interference patterns, diffraction patterns, and thermodynamics, this tool has you covered.

## Features

### 1. Mathematical Functions
- Plot predefined mathematical functions or enter your own custom functions.
- Choose the color for your plot.
- Download the generated plots.

### 2. Waveform Generation
- Generate sine waveforms with adjustable sampling frequency, frequency, and duration.
- Add noise to the waveform if desired.
- Perform and visualize Fast Fourier Transform (FFT) analysis.
- Download the generated waveform and FFT plots.

### 3. Data Upload and Analysis
- Upload CSV files for analysis.
- View statistical summaries and correlation matrices of your data.
- Perform data aggregation, filtering, and transformation.
- Generate various visualizations such as scatter plots, line charts, bar charts, pie charts, histograms, box plots, and pair plots.
- Download visualizations as images.
- Use Linear Regression on the uploaded data.

### 4. Solve Equation
- Enter mathematical equations to find their roots.
- Visualize the function and its roots in an interactive plot.
- Download the plot showing the function and its roots.

### 5. Interference Patterns
- Generate interference patterns based on wavelength, slit distance, and screen distance.
- Download the interference pattern plot.

### 6. Diffraction Patterns
- Generate diffraction patterns based on wavelength, slit width, and screen distance.
- Download the diffraction pattern plot.

### 7. Thermodynamics
- Visualize temperature change over time based on initial temperature, ambient temperature, and cooling rate.
- Download the thermodynamics plot.

## Installation

To run this application, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

2. Navigate to the project directory:
    ```bash
    cd your-repository
    ```

3. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Requirements

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- SciPy

## Usage

1. Open the Streamlit application in your web browser.
2. Use the sidebar to select the type of data or functionality you want to explore.
3. Follow the interactive widgets to customize and generate your visualizations or analyses.
4. Download any generated plots or visualizations as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the framework used to build the app.
- [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/), and [SciPy](https://scipy.org/) for the scientific computing libraries used.
- [Scikit-learn](https://scikit-learn.org/) for implementing Linear Regression with performance metrics.

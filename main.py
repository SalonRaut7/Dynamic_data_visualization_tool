import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
from scipy.optimize import fsolve
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Dynamic Data Visualization Tool")

# Sidebar for user input
st.sidebar.header("Choose the Type of Data")

option = st.sidebar.selectbox(
    "Select the type of data:",
    ["Mathematical Functions", "Waveform", "Upload Data", "Solve Equation", "Interference Patterns", "Diffraction Patterns", "Thermodynamics"]
)

# Function to plot mathematical functions
def plot_math_functions():
    st.subheader("Mathematical Functions")
    
    # Dropdown for predefined functions
    function_option = st.selectbox(
        "Choose a mathematical function:",
        ["x**2", "np.sin(x)", "np.cos(x)", "np.tan(x)", "np.exp(x)", "np.log(x+1)"]
    )
    
    # Input for custom function
    custom_func = st.text_input("Or enter a custom function (use 'x' as variable):")
    
    # Use predefined function or custom function
    func = function_option if not custom_func else custom_func
    
    # Preprocess custom function input
    if 'sin' in func or 'cos' in func or 'tan' in func or 'exp' in func or 'log' in func:
        if not func.startswith("np."):
            func = "np." + func

    color = st.color_picker("Pick a color for the plot", "#00f900")
    
    if func:
        try:
            x = np.linspace(-10, 10, 400)
            y = eval(func, {"np": np, "x": x})  # Evaluate function with NumPy
            plt.figure(figsize=(12, 6))
            plt.plot(x, y, color=color)
            plt.title(f'Plot of {func}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            
            # Save and display plot
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            st.image(buf, caption="Mathematical Function Plot")
            st.download_button(label="Download Plot", data=buf, file_name='math_function_plot.png', mime='image/png')
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to plot waveform
def plot_waveform():
    st.subheader("Waveform")
    fs = st.number_input("Enter sampling frequency (Hz):", min_value=1, value=1000)
    f = st.number_input("Enter frequency of the sine wave (Hz):", min_value=1, value=5)
    duration = st.number_input("Enter duration of the waveform (s):", min_value=1, value=1)
    add_noise = st.checkbox("Add noise to waveform")

    if st.button("Generate Waveform"):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * f * t)
        if add_noise:
            noise = np.random.normal(0, 0.1, waveform.shape)
            waveform += noise
        
        plt.figure(figsize=(12, 6))
        plt.plot(t, waveform)
        plt.title('Waveform of a Sine Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="Waveform Plot")
        st.download_button(label="Download Plot", data=buf, file_name='waveform_plot.png', mime='image/png')

        # FFT Analysis
        fft_vals = np.fft.fft(waveform)
        fft_freq = np.fft.fftfreq(len(waveform), 1/fs)
        
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq, np.abs(fft_vals))
        plt.title('Frequency Domain of the Waveform')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="FFT Plot")
        st.download_button(label="Download FFT Plot", data=buf, file_name='fft_plot.png', mime='image/png')


def upload_file():
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of the dataset:")
        st.write(data.head())

        # Statistical Information
        st.subheader("Statistical Information")
        st.write("Summary Statistics:")
        st.write(data.describe(include='all'))
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        if st.checkbox("Show Correlation Matrix"):
            # Filter numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                corr = numeric_data.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
                plt.title('Correlation Matrix')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf, caption="Correlation Matrix")
                st.download_button(label="Download Correlation Matrix", data=buf, file_name='correlation_matrix.png', mime='image/png')
            else:
                st.warning("No numeric columns available for correlation analysis.")
        
        # Data Aggregation
        st.subheader("Data Aggregation")
        if not data.empty:
            group_col = st.selectbox("Select column to group by:", data.columns)
            if group_col:
                # Filter numeric columns only for aggregation
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if not numeric_columns.empty:
                    agg_func = st.selectbox("Select aggregation function:", ["mean", "sum", "median", "std", "min", "max"])
                    if st.button("Apply Aggregation"):
                        try:
                            # Ensure the group column is not used for aggregation
                            if group_col in numeric_columns:
                                agg_columns = numeric_columns.difference([group_col])
                            else:
                                agg_columns = numeric_columns

                            if not agg_columns.empty:
                                agg_data = data.groupby(group_col)[agg_columns].agg(agg_func)
                                st.write(f"Aggregated Data ({agg_func}):")
                                st.write(agg_data)
                            else:
                                st.warning("No numeric columns available for aggregation.")
                        except Exception as e:
                            st.error(f"An error occurred during aggregation: {e}")
                else:
                    st.warning("No numeric columns available for aggregation.")
        
        # Data Filtering
        st.subheader("Data Filtering")
        if not data.empty:
            filter_col = st.selectbox("Select column to filter by:", data.columns)
            if filter_col:
                if pd.api.types.is_numeric_dtype(data[filter_col]):
                    # Numeric column
                    filter_value = st.number_input(f"Enter value for filtering {filter_col}:", min_value=float(data[filter_col].min()), max_value=float(data[filter_col].max()))
                    if st.button("Apply Filter"):
                        try:
                            filtered_data = data[data[filter_col] == filter_value]
                            st.write(f"Filtered Data (where {filter_col} = {filter_value}):")
                            st.write(filtered_data)
                        except Exception as e:
                            st.error(f"An error occurred during filtering: {e}")
                else:
                    # Non-numeric column
                    filter_value = st.text_input(f"Enter value for filtering {filter_col}:")
                    if st.button("Apply Filter"):
                        try:
                            filtered_data = data[data[filter_col].astype(str) == filter_value]
                            st.write(f"Filtered Data (where {filter_col} = {filter_value}):")
                            st.write(filtered_data)
                        except Exception as e:
                            st.error(f"An error occurred during filtering: {e}")

        # Data Transformation
        st.subheader("Data Transformation")
        if not data.empty:
            transform_col = st.selectbox("Select column to transform:", data.columns)
            if transform_col:
                transform_option = st.selectbox("Select transformation:", ["None", "Normalize", "Standardize"])
                if st.button("Apply Transformation"):
                    try:
                        if pd.api.types.is_numeric_dtype(data[transform_col]):
                            if transform_option == "Normalize":
                                data[transform_col] = (data[transform_col] - data[transform_col].min()) / (data[transform_col].max() - data[transform_col].min())
                            elif transform_option == "Standardize":
                                data[transform_col] = (data[transform_col] - data[transform_col].mean()) / data[transform_col].std()
                            st.write(f"Transformed Data ({transform_option}):")
                            st.write(data)
                        else:
                            st.warning(f"The selected column ({transform_col}) contains non-numeric values. Transformation can only be applied to numeric columns.")
                    except Exception as e:
                        st.error(f"An error occurred during transformation: {e}")

        # Visualization Options
        st.write("Choose the visualization type:")
        vis_option = st.selectbox(
            "Select visualization type:",
            ["None", "Scatter Plot", "Line Chart", "Bar Chart", "Pie Chart", "Histogram", "Box Plot", "Pair Plot"]
        )
        
        if vis_option == "Scatter Plot":
            st.subheader("Scatter Plot")
            x_col = st.selectbox("Select X-axis column:", data.columns)
            y_col = st.selectbox("Select Y-axis column:", data.columns)
            if st.button("Generate Scatter Plot"):
                try:
                    fig = px.scatter(data, x=x_col, y=y_col, title=f'Scatter Plot of {x_col} vs {y_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating scatter plot: {e}")

        elif vis_option == "Line Chart":
            st.subheader("Line Chart")
            x_col = st.selectbox("Select X-axis column:", data.columns)
            y_col = st.selectbox("Select Y-axis column:", data.columns)
            if st.button("Generate Line Chart"):
                try:
                    fig = px.line(data, x=x_col, y=y_col, title=f'Line Chart of {y_col} over {x_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating line chart: {e}")

        elif vis_option == "Bar Chart":
            st.subheader("Bar Chart")
            x_col = st.selectbox("Select X-axis column:", data.columns)
            y_col = st.selectbox("Select Y-axis column:", data.columns)
            if st.button("Generate Bar Chart"):
                try:
                    fig = px.bar(data, x=x_col, y=y_col, title=f'Bar Chart of {y_col} by {x_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating bar chart: {e}")

        elif vis_option == "Pie Chart":
            st.subheader("Pie Chart")
            values_col = st.selectbox("Select values column:", data.columns)
            names_col = st.selectbox("Select names column:", data.columns)
            if st.button("Generate Pie Chart"):
                try:
                    fig = px.pie(data, values=values_col, names=names_col, title=f'Pie Chart of {values_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating pie chart: {e}")

        elif vis_option == "Histogram":
            st.subheader("Histogram")
            x_col = st.selectbox("Select column for histogram:", data.columns)
            if st.button("Generate Histogram"):
                try:
                    fig = px.histogram(data, x=x_col, title=f'Histogram of {x_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating histogram: {e}")

        elif vis_option == "Box Plot":
            st.subheader("Box Plot")
            x_col = st.selectbox("Select column for box plot:", data.columns)
            if st.button("Generate Box Plot"):
                try:
                    fig = px.box(data, y=x_col, title=f'Box Plot of {x_col}')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating box plot: {e}")

        elif vis_option == "Pair Plot":
            st.subheader("Pair Plot")
            if st.button("Generate Pair Plot"):
                try:
                    fig = px.scatter_matrix(data, title='Pair Plot')
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating pair plot: {e}")
        st.subheader("Predictive Analytics")
        if st.checkbox("Perform Predictive Analytics"):
            st.info("For simplicity, we'll implement a basic linear regression model.")
            target = st.selectbox("Select the target column:", data.columns)
            features = st.multiselect("Select feature columns:", data.columns.drop(target))
    
        if st.button("Train Model"):
            if not features or not target:
                st.error("Please select at least one feature and a target.")
            else:
            # Check if the selected features and target are numeric
                non_numeric_cols = [col for col in features + [target] if not pd.api.types.is_numeric_dtype(data[col])]
            
                if non_numeric_cols:
                    st.error(f"The following selected columns are non-numeric: {', '.join(non_numeric_cols)}. Linear regression can only be applied to numeric columns.")
                else:
                    try:
                        X = data[features]
                        y = data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        
                        st.write("Model Performance:")
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")
                        
                        # Plotting the true vs predicted values
                        fig = plt.figure(figsize=(10, 6))
                        plt.scatter(y_test, y_pred, color='blue')
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
                        plt.title("True vs Predicted Values")
                        plt.xlabel("True Values")
                        plt.ylabel("Predicted Values")
                        plt.grid(True)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")


def solve_equation():
    st.subheader("Solve Equation")
    
    # Input for the equation
    equation = st.text_input("Enter the equation (use 'x' as variable, e.g., x**2 - 4):")
    
    if st.button("Find Roots"):
        if equation:
            try:
                # Define a function based on the equation
                def func(x):
                    return eval(equation, {"np": np, "x": x})
                
                # Guess initial values for the roots
                guesses = np.linspace(-10, 10, 5)
                roots = []
                for guess in guesses:
                    root = fsolve(func, guess)[0]
                    if all(abs(root - r) > 1e-5 for r in roots):  # Check if root is unique
                        roots.append(root)
                
                # Display roots in an interactive table
                roots_df = pd.DataFrame(roots, columns=["Root"])
                st.subheader("Roots of the Equation")
                st.dataframe(roots_df)
                
                # Create a Plotly figure to visualize the function and its roots
                x = np.linspace(-10, 10, 400)
                y = func(x)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function'))
                fig.add_trace(go.Scatter(x=roots, y=[0]*len(roots), mode='markers', name='Roots', marker=dict(color='red', size=10)))
                fig.update_layout(title='Equation and Roots', xaxis_title='x', yaxis_title='f(x)')
                
                st.subheader("Function Plot with Roots")
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

def plot_interference_pattern():
    st.subheader("Interference Patterns")
    
    # Inputs for the interference pattern
    wavelength = st.number_input("Enter wavelength of light (in nm):", min_value=1, value=500)
    slit_distance = st.number_input("Enter distance between slits (in mm):", min_value=0.1, value=1.0)
    screen_distance = st.number_input("Enter distance to the screen (in m):", min_value=0.1, value=1.0)
    
    if st.button("Generate Interference Pattern"):
        # Constants
        slit_distance_m = slit_distance * 1e-3  # Convert mm to meters
        
        # Generate x values
        x = np.linspace(-0.05, 0.05, 1000)
        
        # Calculate interference pattern
        k = 2 * np.pi / (wavelength * 1e-9)  # Wavenumber
        pattern = np.sin(k * slit_distance_m * x / screen_distance)**2
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, pattern)
        plt.title('Interference Pattern')
        plt.xlabel('Position on Screen (m)')
        plt.ylabel('Intensity')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="Interference Pattern Plot")
        st.download_button(label="Download Plot", data=buf, file_name='interference_pattern.png', mime='image/png')

def plot_diffraction_pattern():
    st.subheader("Diffraction Patterns")
    
    # Inputs for the diffraction pattern
    wavelength = st.number_input("Enter wavelength of light (in nm):", min_value=1, value=500)
    slit_width = st.number_input("Enter width of the slit (in mm):", min_value=0.1, value=1.0)
    screen_distance = st.number_input("Enter distance to the screen (in m):", min_value=0.1, value=1.0)
    
    if st.button("Generate Diffraction Pattern"):
        # Constants
        slit_width_m = slit_width * 1e-3  # Convert mm to meters
        
        # Generate x values
        x = np.linspace(-0.05, 0.05, 1000)
        
        # Calculate diffraction pattern
        k = 2 * np.pi / (wavelength * 1e-9)  # Wavenumber
        pattern = (np.sin(k * slit_width_m * x / screen_distance) / (k * slit_width_m * x / screen_distance))**2
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, pattern)
        plt.title('Diffraction Pattern')
        plt.xlabel('Position on Screen (m)')
        plt.ylabel('Intensity')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="Diffraction Pattern Plot")
        st.download_button(label="Download Plot", data=buf, file_name='diffraction_pattern.png', mime='image/png')

def plot_thermodynamics():
    st.subheader("Thermodynamics")
    
    # Inputs for the thermodynamics plot
    initial_temp = st.number_input("Enter initial temperature (in °C):", min_value=-100, value=25)
    ambient_temp = st.number_input("Enter ambient temperature (in °C):", min_value=-100, value=25)
    time = st.number_input("Enter time (in seconds):", min_value=0, value=60)
    cooling_rate = st.number_input("Enter cooling rate (°C/s):", min_value=0.01, value=0.1)
    
    if st.button("Generate Thermodynamics Plot"):
        t = np.linspace(0, time, 1000)
        temperature = ambient_temp + (initial_temp - ambient_temp) * np.exp(-cooling_rate * t)
        
        plt.figure(figsize=(12, 6))
        plt.plot(t, temperature)
        plt.title('Temperature vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.image(buf, caption="Thermodynamics Plot")
        st.download_button(label="Download Plot", data=buf, file_name='thermodynamics_plot.png', mime='image/png')

# Display the selected option
if option == "Mathematical Functions":
    plot_math_functions()
elif option == "Waveform":
    plot_waveform()
elif option == "Upload Data":
    upload_file()
elif option == "Solve Equation":
    solve_equation()
elif option == "Interference Patterns":
    plot_interference_pattern()
elif option == "Diffraction Patterns":
    plot_diffraction_pattern()
elif option == "Thermodynamics":
    plot_thermodynamics()
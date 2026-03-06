# Real Estate Price Prediction

A machine learning project designed to predict apartment prices across various cities in Poland using a neural network regression model.

## Features

- **Automated Data Processing**: Loads and combines multiple CSV datasets from specified directories.
- **Data Cleaning**: Handles missing values, drops irrelevant features, and cleans categorical data.
- **Preprocessing Pipeline**: Implements `StandardScaler` for numerical features and `OneHotEncoder` for categorical variables.
- **Deep Learning Model**: Utilizes a TensorFlow/Keras `Sequential` neural network with ELU activation layers for regression.
- **Performance Visualization**: Generates charts for Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), and error distribution.
- **Scenario Analysis**: Predicts prices for predefined apartment scenarios

## Technologies Used

- **Python**: Core programming language.
- **TensorFlow / Keras**: Deep learning framework for building the neural network.
- **Scikit-learn**: Data preprocessing pipelines and model evaluation tools.
- **Pandas & NumPy**: Data manipulation and numerical computations.
- **Matplotlib**: Visualization of model performance and errors.

## Performance

The model evaluation includes:
- **MAPE & MAE tracking** during training and validation.
- **Error Percentage Distribution** (Histogram and Boxplot) to identify prediction accuracy and outliers.
- **Quantile Statistics** for error analysis.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib
   ```
2. **Setup Data**:
   Place your apartment data CSV files in a directory and update the `directory_path` in `main.py`.
3. **Run the Script**:
   ```bash
   python main.py
   ```

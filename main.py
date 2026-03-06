import os
import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout
from keras.src.utils.module_utils import tensorflow
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Load all CSV files from a directory
def load_data_from_directory(directory_path):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Data cleaning and preprocessing
def clean_data(df):
    # Drop irrelevant features
    columns_to_drop = ['id', 'latitude', 'longitude', 'buildingMaterial']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Fill missing values
    if 'condition' in df.columns:
        df['condition'] = df['condition'].fillna('standard')

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('unknown')

    # Check if any columns still have missing values
    remaining_nulls = df.isnull().sum()
    if remaining_nulls.any():
        print("Remaining columns with missing values after fill:")
        print(remaining_nulls[remaining_nulls > 0])

    # Drop rows with missing values in critical columns
    critical_columns = ['squareMeters', 'rooms', 'floor', 'floorCount', 'price']
    print(f"Dropping rows with missing values in critical columns: {critical_columns}")
    initial_shape = df.shape
    df = df.dropna(subset=critical_columns)
    print(f"Rows dropped: {initial_shape[0] - df.shape[0]}")

    # Encode binary columns
    binary_columns = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    mapping = {'yes': 1.0, 'no': -1.0, 'unknown': 0.0}
    for col in binary_columns:
        if col in df.columns:
            # Count occurrences of each value
            yes_count = (df[col] == 'yes').sum()
            no_count = (df[col] == 'no').sum()
            unknown_count = (df[col] == 'unknown').sum()
            print(f"Column '{col}': {yes_count} 'yes', {no_count} 'no', {unknown_count} 'unknown'")
            df[col] = df[col].map(mapping)
    return df

# Load and clean data
directory_path = os.path.join(os.path.dirname(__file__), "data")
data = load_data_from_directory(directory_path)
data = clean_data(data)

# Split features and target
X = data.drop(columns=['price'])
y = data['price']

categorical_columns = ['city', 'type', 'ownership', 'condition']
numerical_columns = ['squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear',
                     'centreDistance', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance',
                     'kindergartenDistance', 'restaurantDistance', 'collegeDistance', 'pharmacyDistance']

# Setup preprocessing
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Combine preprocessing steps into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Data split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_preprocessed = pipeline.fit_transform(X_train)
X_valid_preprocessed = pipeline.transform(X_valid)
X_test_preprocessed = pipeline.transform(X_test)

# Set seeds for reproducibility
tensorflow.random.set_seed(41654)
tensorflow.keras.utils.set_random_seed(35648)

# Neural Network Architecture
model = Sequential([
    Dense(30, activation='elu'),
    Dense(30, activation='elu'),
    Dense(30, activation='elu'),
    Dense(30, activation='elu'),
    Dense(30, activation='elu'),
    Dense(1)
])

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name='adam',
    ),
    loss=tensorflow.keras.losses.MeanAbsolutePercentageError(),
    metrics=['mae']
)

# Training
callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(X_train_preprocessed, y_train, epochs=500, batch_size=8, validation_data=(X_valid_preprocessed, y_valid), callbacks=[callback])


"""
# Train the model
history = model.fit(
    X_train_preprocessed, 
    y_train, 
    validation_split=0.2, 
    epochs=50, 
    batch_size=32
)
"""

# Evaluate results
test_loss, test_mae = model.evaluate(X_test_preprocessed, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Visualize training progress
fig = plt.figure(figsize=(15, 5))

# Plot Mean Absolute Percentage Error
ax = fig.add_subplot(1, 2, 1)
ax.set_title("Mean Absolute Percentage Error")
ax.plot(history.history['loss'], label='Train Data')
ax.plot(history.history['val_loss'], label='Validation Data')
ax.set_ylim(top=15)
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.set_title("Mean Absolute Error")
ax.plot(history.history['mae'], label='Train Data')
ax.plot(history.history['val_mae'], label='Validation Data')
ax.set_ylim(top=120000)
ax.legend()

plt.show()


# Error details
test_input = X_train_preprocessed
test_output = y_train

y_pred = model.predict(test_input)

testResult = pd.DataFrame(test_input).copy()
testResult['y_test'] = test_output.reset_index(drop=True)
testResult['y_pred'] = y_pred.flatten().round()
testResult['error'] = 100 * (testResult['y_pred'] - testResult['y_test']) / testResult['y_test']
testResult['abs_error'] = abs(testResult['error'])

quantiles = testResult['error'].quantile([0.25, 0.5, 0.75]).to_numpy()

def percent_formatter(x, pos):
    return f"{round(x)}%"

fig = plt.figure(figsize=(20, 5))

# Error distribution plots
ax = fig.add_subplot(1, 2, 1)
ax.set_title("Error Percentage Distribution")
ax.hist(testResult['error'], bins=100)
ax.axvline(quantiles[0], color='k', linestyle='dashed', linewidth=1)
ax.axvline(quantiles[1], color='k', linestyle='solid', linewidth=1)
ax.axvline(quantiles[2], color='k', linestyle='dashed', linewidth=1)
ax.xaxis.set_major_formatter(percent_formatter)
ax.yaxis.set_visible(False)

# Boxplot of Error Percentage Distribution
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Error Percentage Distribution (Boxplot)")
ax.boxplot([testResult['error']], showfliers=False, vert=False)
ax.set_xlim(-50, 50)
ax.set_xticks(np.arange(-50, 51, 10))
ax.grid(linewidth=1, color="#ccc", which='major')
ax.yaxis.set_visible(False)
ax.xaxis.set_major_formatter(percent_formatter)

plt.show()

# Error Statistics
errorStats = pd.DataFrame(testResult['error'].quantile([0.25, 0.5, 0.75])).reset_index().rename({'index': 'percentile'}, axis=1)
errorStats['percentile'] = errorStats['percentile'] * 100
errorStats['percentile'] = errorStats['percentile'].astype(int)
errorStats['error'] = round(errorStats['error'] * 10) / 10

print("Error Statistics:")
print(errorStats)

# Predict prices for specific scenarios
baseline = {
    'city': 'krakow',
    'type': 'blockOfFlats',
    'squareMeters': 44.0,
    'rooms': 2.0,
    'floor': 1.0,
    'floorCount': 4.0,
    'buildYear': 2000.0,
    'centreDistance': 4.0,
    'poiCount': 30.0,
    'schoolDistance': 10.0,
    'clinicDistance': 10.0,
    'postOfficeDistance' : 10.0,
    'kindergartenDistance' : 10.0,
    'restaurantDistance' : 10.0,
    'collegeDistance' : 10.0,
    'pharmacyDistance' : 10.0,
    'condition': 'standard',
    'hasParkingSpace': -1.0,
    'hasBalcony': -1.0,
    'hasElevator': -1.0,
    'hasSecurity': -1.0,
    'hasStorageRoom': -1.0
}

# Ensure all required columns are present in the input data
required_columns = X_train.columns
for col in required_columns:
    if col not in baseline:
        baseline[col] = 0 if col in numerical_columns else 'unknown'

# Create variations of the baseline scenario
scenarios = {
    "SizePlus": baseline.copy(),
    "OldTenement": baseline.copy(),
    "CityCentre": baseline.copy(),
    "FullAmenities": baseline.copy(),
    "Penthouse": baseline.copy(),
    "NewConstruction": baseline.copy(),
    "LuxuryApartment": baseline.copy(),
    "LowBudgetApartment": baseline.copy(),
    "NewConstructionWarszawa": baseline.copy(),
    "CityCentreFar": baseline.copy(),
}

# Modifying scenarios
scenarios["SizePlus"]['squareMeters'] = 65
scenarios["SizePlus"]['rooms'] = 3

scenarios["OldTenement"]['type'] = 'tenement'
scenarios["OldTenement"]['buildYear'] = 1923

scenarios["NewConstruction"]['buildYear'] = 2023
scenarios["NewConstruction"]['condition'] = 'premium'

scenarios["NewConstructionWarszawa"]['city'] = 'warszawa'
scenarios["NewConstructionWarszawa"]['buildYear'] = 2023
scenarios["NewConstructionWarszawa"]['condition'] = 'premium'

scenarios["CityCentre"]['centreDistance'] = 0.5
scenarios["CityCentreFar"]['centreDistance'] = 50.0

scenarios["FullAmenities"]['hasParkingSpace'] = 1.0
scenarios["FullAmenities"]['hasBalcony'] = 1.0
scenarios["FullAmenities"]['hasElevator'] = 1.0
scenarios["FullAmenities"]['hasSecurity'] = 1.0
scenarios["FullAmenities"]['hasStorageRoom'] = 1.0

scenarios["Penthouse"]['floor'] = 10
scenarios["Penthouse"]['floorCount'] = 10
scenarios["Penthouse"]['hasBalcony'] = 1.0
scenarios["Penthouse"]['hasElevator'] = 1.0
scenarios["Penthouse"]['hasSecurity'] = 1.0
scenarios["Penthouse"]['hasStorageRoom'] = 1.0

scenarios["LuxuryApartment"]['squareMeters'] = 80
scenarios["LuxuryApartment"]['rooms'] = 4
scenarios["LuxuryApartment"]['hasParkingSpace'] = 1.0
scenarios["LuxuryApartment"]['hasBalcony'] = 1.0
scenarios["LuxuryApartment"]['hasElevator'] = 1.0
scenarios["LuxuryApartment"]['hasSecurity'] = 1.0
scenarios["LuxuryApartment"]['hasStorageRoom'] = 1.0
scenarios["LuxuryApartment"]['condition'] = 'premium'

scenarios["LowBudgetApartment"]['squareMeters'] = 30
scenarios["LowBudgetApartment"]['rooms'] = 1
scenarios["LowBudgetApartment"]['condition'] = 'low'

# Convert dictionary to DataFrame
inputData = pd.DataFrame.from_dict(scenarios, orient='index')

# Reorder columns to match the training data
inputData = inputData[required_columns]

# Preprocess the input data
inputData_preprocessed = pipeline.transform(inputData)

# Predict prices for each scenario
inputData['price'] = model.predict(inputData_preprocessed)
inputData['price'] = (inputData['price'] / 100).round() * 100

# Display results in a readable format
pd.set_option('display.max_columns', None)
print(inputData.transpose())
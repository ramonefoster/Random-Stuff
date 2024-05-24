import pandas as pd
import csv
import joblib
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import keras
from keras import layers
from scipy.stats import uniform

import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

RNA_CSV_PATH = 'data/dataset.csv'
RNA_TRAINTEST_PATH = 'data/rf_train_test.pkl'
RNA_SCALER_PATH = 'data/rf_scaler_test.pkl'
RNA_MODEL_PATH = 'data/rf_model.keras'

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180.0


def add_noise(df, noise_level=0.01):
    noisy_df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Only add noise to numerical columns
            noise = np.random.normal(0, noise_level, df[col].shape)
            noisy_df[col] += noise
    return noisy_df

def scale_features(df, scale_factor=0.05):
    scaled_df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Only scale numerical columns
            scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
            scaled_df[col] *= scale
    return scaled_df

def shift_features(df, shift_factor=0.01):
    shifted_df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Only shift numerical columns
            shift = np.random.uniform(-shift_factor, shift_factor)
            shifted_df[col] += shift
    return shifted_df

def check_format(input):
    """
    Check the format of a given input and extract components.

    The expected format is 'x1:x2:x3' or 'x1 x2 x3',
    where 'x1', 'x2', and 'x3' are separated by a colon (':') or a space (' ').

    Parameters:
        input (str): The input to be checked and parsed.

    Returns:
        False: If the input does not match the expected format.
        list: A list containing the three components extracted from the input if the format is correct.
    """
    separators = [':', ' ']
    separator = None
    for sep in separators:
        if sep in input:
            separator = sep
            break

    if separator is None:
        return False

    components = input.split(separator)

    # Check for correct format
    if len(components) != 3:
        return False
    else:
        return components
    
def hms_to_hours(time_string):
    """
    Converts Hours string to float
    :param time_string: Hours String (hh:mm:ss.ss)
    """        
    # Verify separator
    try:
        components = check_format(time_string)

        if components:
            hours = abs(int(components[0]))
            minutes = int(components[1])
            seconds = float(components[2])

            total_hours = hours + minutes / 60 + seconds / 3600

            sign = -1 if "-" in time_string else 1
            return sign*total_hours
        else:
            return None
    except:
        return None

def dms_to_degrees(degrees_string):
    """
    Converts Degrees string to float
    :param degrees_string: Degrees String (dd:mm:ss.ss)
    """
    # Verify separator
    try:
        components = check_format(degrees_string)

        if components:
            degrees_int = abs(int(components[0]))
            minutes = int(components[1])    
            seconds = float(components[2])

            degrees = degrees_int + minutes / 60 + seconds / 3600

            sign = -1 if "-" in degrees_string else 1
            return sign*degrees
        else:
            return None
    except:
        return None

def is_numeric(input):
    """
    Check if the input is a numeric value.
    Parameters:
        input (int or float): The value to be checked.
    Returns:
        bool: True if the input is numeric (int or float), False otherwise.
    """
    if isinstance(input, (int, float)):
        return True
    else:
        return False
         
def check_exists(path):
    """
    Check if a file or directory exists at the given path.
    Parameters:
        path (str): The path to check for existence.
    Returns:
        bool: True if a file or directory exists at the given path, False otherwise.
    """
    import os
    if os.path.exists(path):
        return True
    else:
        return False

def hours_to_hms(hours, decimal_digits=0):
    """
    Converts Float Hour to string Hour, in format hh:mm:ss:cc
    :param hours: Hours (float)
    """
    if is_numeric(hours):        
        sign = "-" if hours < 0 else ""
        hours = abs(hours)
        whole_hours = int(hours)
        fractional_hours = hours - whole_hours

        minutes = int(fractional_hours * 60)
        fractional_minutes = fractional_hours * 60 - minutes

        seconds = int(fractional_minutes * 60)
        fractional_seconds = fractional_minutes * 60 - seconds

        seconds_str = f"{seconds:02}.{int(fractional_seconds * (10 ** decimal_digits)):02d}"

        time_string = f"{sign}{whole_hours:02}:{minutes:02}:{seconds_str}"
        
        return time_string
    else:
        return None

def degrees_to_dms(degrees):
    """
    Converts Degrees to string, in format dd:mm:ss:cc
    :param hours: Degrees (float)
    """
    if is_numeric(degrees):
        sign = "-" if degrees < 0 else "+"
        degrees = abs(degrees)
        degrees_int = int(degrees)
        minutes = int((degrees - degrees_int) * 60)
        seconds = int(((degrees - degrees_int) * 60 - minutes) * 60)
        seconds_decimal = int((((degrees - degrees_int) * 60 - minutes) * 60 - seconds) * 100)

        # Formated value
        degrees_string = f'{sign}{degrees_int:02}:{minutes:02}:{seconds:02}.{seconds_decimal:02}'

        return degrees_string
    else:
        return None
    
def get_elevation_azimuth(ha, dec, latitude):
    """Calculates Azimuth and Elevation
    params: HourAngle, Declination, Latitude
    return: Elevation, Azimuth
    """ 
    if not is_numeric(ha):
        ha = hms_to_hours(ha)
    if not is_numeric(dec):
        dec = dms_to_degrees(dec)
    if not is_numeric(latitude):
        latitude = dms_to_degrees(latitude)

    H = ha * 15

    #altitude calc
    sinAltitude = (math.sin(dec * DEG2RAD)) * (math.sin(latitude * DEG2RAD)) + (math.cos(dec * DEG2RAD) * math.cos(latitude * DEG2RAD) * math.cos(H * DEG2RAD))
    elevation = math.asin(sinAltitude) * RAD2DEG #altura em graus
    elevation = round(elevation, 2)

    #azimuth calc
    y = -1 * math.sin(H * DEG2RAD)
    x = (math.tan(dec * DEG2RAD) * math.cos(latitude * DEG2RAD)) - (math.cos(H * DEG2RAD) * math.sin(latitude * DEG2RAD))

    #This AZposCalc is the initial AZ for dome positioning
    azimuth = math.atan2(y, x) * RAD2DEG

    #converting neg values to pos
    if (azimuth < 0) :
        azimuth = azimuth + 360    

    return elevation, azimuth

class CSVHandler():
    @staticmethod
    def read_data():
        """
        Read the CSV file and extract the variables of interest.

        Returns:
            X (pandas.DataFrame): Dataframe containing the selected input variables.
            Y (pandas.DataFrame): Dataframe containing the selected output variables.
        """
        if not check_exists(RNA_CSV_PATH):
            return None, None
        
        dataframe = pd.read_csv(RNA_CSV_PATH)

        X = dataframe.loc[:, ["azimuth", "elevation", "ah_star", "dec_star", "prev_ha", "prev_dec"]]
        Y = dataframe[["err_ah", "err_dec"]]

        # Create new columns using .loc to avoid the SettingWithCopyWarning
        X.loc[:, "dist_ha"] = X["ah_star"] - X["prev_ha"]
        X.loc[:, "dist_dec"] = X["dec_star"] - X["prev_dec"]
        
        X.loc[:, "pier_side"] = X["ah_star"] > 0


        return X, Y

    @staticmethod
    def create_file():
        """
        Create a new CSV file with a header row.

        """
        headerList = ['ah_star', 'dec_star', 'ah_scope', 'dec_scope', 'err_ah', 'err_dec', 'elevation', 'azimuth', 'temperature', 'prev_ha', 'prev_dec']

        with open(RNA_CSV_PATH, 'w') as file:
            dw = csv.DictWriter(file, delimiter=',', 
                                fieldnames=headerList)
            dw.writeheader()
            file.close()

    @staticmethod
    def save_dataframe(ah_star, dec_star, ah_scope, dec_scope, azimuth, elevation, temperature, prev_ha, prev_dec):
        """
        Save data to a CSV file.

        Parameters:
            ah_star (float): Real star hour angle.
            dec_star (float): Real star declination.
            ah_scope (float): Telescope hour angle.
            dec_scope (float): Telescope declination.
            azimuth (float): Telescope azimuth.
            elevation (float): Telescope elevation.
            temperature (float): Internal dome temperature.
            prev_ha (float): Telescope HA position before slewing to target
            prev_dec (float): Telescope DEC position before slewing to target

        """
        if not check_exists(RNA_CSV_PATH):
            CSVHandler.create_file()
        
        err_ah = ah_star - ah_scope
        err_dec = dec_star - dec_scope

        d = {'ah_star': [ah_star], 'dec_star': [dec_star],
            'ah_scope': [ah_scope], 'dec_scope': [dec_scope],
            'err_ah': [err_ah], 'err_dec': [err_dec], 
            'elevation':[elevation], 'azimuth': [azimuth], 
            'temperature': [temperature], 'prev_ha': [prev_ha], 'prev_dec': [prev_dec] }

        df = pd.DataFrame.from_dict(data=d)
        df.to_csv((RNA_CSV_PATH), mode='a', index=False, header=False)

class RandomForest():
    @staticmethod
    def create_model(neurons=64, dropout_rate=0.2, optimizer='adam'):
        model = keras.Sequential()
        model.add(layers.Dense(neurons, activation='relu', input_dim=80))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(neurons//2, activation='relu'))
        model.add(layers.Dense(2))  # Output layer
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    
    @staticmethod
    def train():
        result = { }
        """
        Perform training of the neural network and save the generated model to a pickle file.

        Returns:
            float: The training score rounded to 2 decimal places.

        """       
        X, Y = CSVHandler.read_data()      

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()

        # Fit and transform training data
        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train)

        # Transform testing data
        X_test_scaled = scaler_X.transform(X_test)
        Y_test_scaled = scaler_Y.transform(Y_test)

        scalers = {
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y
        }

        # Save the scalers dictionary
        joblib.dump(scalers, RNA_SCALER_PATH)

        #BEST
        model = keras.Sequential()
        model.add(keras.Input(shape=(X_train.shape[1],)))
        model.add(layers.Dense(256, activation='tanh'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(7, activation='tanh'))
        model.add(layers.Dense(2)) 
        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['acc', 'mse'])
        
        model.save(RNA_MODEL_PATH)

        # Define a checkpoint callback to save the best model during training
        checkpoint_callback = keras.callbacks.ModelCheckpoint(RNA_MODEL_PATH, save_best_only=True)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        epochs = 300
        batch = 50
        # keras_model = KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch)
        # scores = cross_val_score(model, X_train_scaled, Y_train_scaled, cv=5)

        # Train the model
        history = model.fit(X_train_scaled, Y_train_scaled, 
                    validation_data=(X_test_scaled, Y_test_scaled),
                    epochs=epochs, batch_size=batch, callbacks=[checkpoint_callback, early_stopping_callback])

        # Calculate and return the training score
        train_loss = model.evaluate(X_train_scaled, Y_train_scaled, verbose=0)
        test_loss = model.evaluate(X_test_scaled, Y_test_scaled, verbose=0)
        loss1, acc1, mse1 = model.evaluate(X_test, Y_test)

        Y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(Y_test_scaled, Y_pred)
        mse = mean_squared_error(Y_test_scaled, Y_pred)
        r2 = r2_score(Y_test_scaled, Y_pred)


        result[f"RESULT"] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "Cross": 0,
            "MeanCross": np.mean(0)
            }

        print(result)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['acc'], label = 'acc')
        plt.plot(history.history['val_acc'], label='val acc')
        plt.title("acc vs Val_acc")
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.legend()
        plt.show()

        return round(loss1, 2)
    
    @staticmethod
    def make_predict(ha=None, dec=None, temp=None, latitude=None, prev_ha=None, prev_dec=None):
        """
        Make predictions using a trained model.

        Args:
            ha (float): Target's hour angle.
            dec (float): Target's declination.
            temp (float): Temperature in degrees Celsius.
            latitude (float): Latitude of the location.

        Returns:
            tuple: Predicted hour angle and declination.

        """
        if not check_exists(RNA_MODEL_PATH):
            return None, None
        
        # rf = joblib.load(RNA_MODEL_PATH)
        model = keras.models.load_model(RNA_MODEL_PATH)
        scalers = joblib.load(RNA_SCALER_PATH)
        scaler_X = scalers['scaler_X']
        scaler_Y = scalers['scaler_Y']

        az, elevation = get_elevation_azimuth(ha, dec, latitude)

        dist_ha = ha - prev_ha
        dist_dec = dec - prev_dec
        pier_side = ha > 0
      

        X_futuro = np.array([[az, elevation, ha, dec, prev_ha, prev_dec, dist_ha, dist_dec, pier_side]])
        X_futuro_scaled = scaler_X.transform(X_futuro)

        Y_rna_prever_futuro_scaled = model.predict(X_futuro_scaled)
        Y_rna_prever_futuro = scaler_Y.inverse_transform(Y_rna_prever_futuro_scaled)

        fator_correct_ha = Y_rna_prever_futuro[0][0]
        fator_correct_dec = Y_rna_prever_futuro[0][1]

        new_ha = ha - fator_correct_ha
        new_dec = dec - fator_correct_dec

        print(hours_to_hms(new_ha), degrees_to_dms(new_dec))
        return (new_ha, new_dec)
          

# RandomForest.train()
RandomForest.make_predict(ha=hms_to_hours("02 58 30"), dec=dms_to_degrees("-26 51 41"), temp=18, 
                          latitude=dms_to_degrees("-22 32 04"), prev_ha=hms_to_hours("02:21:46.67"), prev_dec=dms_to_degrees("+09 53 32"))


# "03:49:18.41"
# "-40:13:25.87"
import pandas as pd
import csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pickle
import joblib
from bayes_opt import BayesianOptimization

from utils.utilities import check_exists, get_elevation_azimuth, hms_to_hours, hours_to_hms, degrees_to_dms, dms_to_degrees, add_noise

RNA_CSV_PATH = 't40_data/dataset.csv'
RNA_TRAINTEST_PATH = 't40_data/rf_train_test.pkl'
RNA_SCALER_PATH = 't40_data/rf_scaler_test.pkl'
RNA_MODEL_PATH = 't40_data/rf_model.pkl'

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
        # Apply noise to original dataset
        noisy_df = add_noise(dataframe)

        # Concatenate original and noisy data
        augmented_df = pd.concat([dataframe, noisy_df])
    
        augmented_df.loc[:, "ah_star"] = augmented_df["ah_star"] * 15
        augmented_df.loc[:, "prev_ha"] = augmented_df["prev_ha"] * 15
        # Feature engineering: create new columns
        augmented_df.loc[:, "dist_ha"] = augmented_df["ah_star"] - augmented_df["prev_ha"]
        augmented_df.loc[:, "dist_dec"] = augmented_df["dec_star"] - augmented_df["prev_dec"]
        augmented_df.loc[:, "pier_side"] = augmented_df["ah_star"] > 0

        # Trigonometric transformations
        augmented_df['azimuth_sin'] = np.sin(np.radians(augmented_df['azimuth']))
        augmented_df['azimuth_cos'] = np.cos(np.radians(augmented_df['azimuth']))

        augmented_df['elevation_sin'] = np.sin(np.radians(augmented_df['elevation']))
        augmented_df['elevation_cos'] = np.cos(np.radians(augmented_df['elevation']))

        augmented_df['ah_star_sin'] = np.sin(np.radians(augmented_df['ah_star']))
        augmented_df['ah_star_cos'] = np.cos(np.radians(augmented_df['ah_star']))

        augmented_df['dec_star_sin'] = np.sin(np.radians(augmented_df['dec_star']))
        augmented_df['dec_star_cos'] = np.cos(np.radians(augmented_df['dec_star']))

        augmented_df['prev_ha_sin'] = np.sin(np.radians(augmented_df['prev_ha']))
        augmented_df['prev_ha_cos'] = np.cos(np.radians(augmented_df['prev_ha']))

        augmented_df['prev_dec_sin'] = np.sin(np.radians(augmented_df['prev_dec']))
        augmented_df['prev_dec_cos'] = np.cos(np.radians(augmented_df['prev_dec']))

        # Select augmented features (X) and target (Y)
        augmented_X = augmented_df[["azimuth", "elevation", "ah_star", "dec_star", 
                                    "dist_ha", "dist_dec", "pier_side",
                                    "azimuth_sin", "azimuth_cos",
                                    "elevation_sin", "elevation_cos",
                                    "ah_star_sin", "ah_star_cos",
                                    "dec_star_sin", "dec_star_cos",
                                    "prev_ha_sin", "prev_ha_cos",
                                    "prev_dec_sin", "prev_dec_cos"]]
        
        augmented_df["err_ah"] = augmented_df["err_ah"] * 15
        augmented_Y = augmented_df[["err_ah", "err_dec"]]
        

        return augmented_X, augmented_Y

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
    def train(best_params = None):
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

        with open(RNA_TRAINTEST_PATH, mode='wb') as f:
            pickle.dump([X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled], f)
        
        if best_params:
            rf_regressor = RandomForestRegressor(**best_params)
        else:
            rf_regressor = RandomForestRegressor(max_depth=113, max_features=0.65,
                                    max_leaf_nodes=119, n_estimators=41, random_state=42)
        rf_regressor.fit(X_train_scaled, Y_train_scaled)
        joblib.dump(rf_regressor, RNA_MODEL_PATH)
        score = rf_regressor.score(X_train_scaled, Y_train_scaled)
        return round(score, 2)
    
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
        
        model = joblib.load(RNA_MODEL_PATH)
        scalers = joblib.load(RNA_SCALER_PATH)
        scaler_X = scalers['scaler_X']
        scaler_Y = scalers['scaler_Y']

        elevation, az = get_elevation_azimuth(ha, dec, latitude)

        dist_ha = ha - prev_ha
        dist_dec = dec - prev_dec
        pier_side = ha > 0
        azimuth_sin = np.sin(np.radians(az))
        azimuth_cos = np.cos(np.radians(az))

        elevation_sin = np.sin(np.radians(elevation))
        elevation_cos = np.cos(np.radians(elevation))

        ah_star_sin = np.sin(np.radians(ha * 15))
        ah_star_cos = np.cos(np.radians(ha * 15))

        dec_star_sin = np.sin(np.radians(dec))
        dec_star_cos = np.cos(np.radians(dec))

        prev_ha_sin = np.sin(np.radians(prev_ha * 15))
        prev_ha_cos = np.cos(np.radians(prev_ha * 15))

        prev_dec_sin = np.sin(np.radians(prev_dec))
        prev_dec_cos = np.cos(np.radians(prev_dec))
      

        X_futuro = np.array([[az, elevation, ha * 15, dec, 
                                    dist_ha * 15, dist_dec, pier_side,
                                    azimuth_sin, azimuth_cos,
                                    elevation_sin, elevation_cos,
                                    ah_star_sin, ah_star_cos,
                                    dec_star_sin, dec_star_cos,
                                    prev_ha_sin, prev_ha_cos,
                                    prev_dec_sin, prev_dec_cos]])
        X_futuro_scaled = scaler_X.transform(X_futuro)

        Y_rna_prever_futuro_scaled = model.predict(X_futuro_scaled)
        Y_rna_prever_futuro = scaler_Y.inverse_transform(Y_rna_prever_futuro_scaled)

        fator_correct_ha = Y_rna_prever_futuro[0][0]
        fator_correct_dec = Y_rna_prever_futuro[0][1]

        new_ha = ha - fator_correct_ha / 15
        new_dec = dec - fator_correct_dec

        print(hours_to_hms(new_ha), degrees_to_dms(new_dec))
        return (new_ha, new_dec)
    
    @staticmethod
    def tunning_hp():
        """
        Perform hyperparameter tuning for a RandomForestRegressor model using Bayesian Optimization.

        Returns:
            RandomForestRegressor: The best model obtained from hyperparameter tuning.
        """        
        scalers = joblib.load(RNA_SCALER_PATH)
        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()

        X_train, X_test, Y_train, Y_test = joblib.load(RNA_TRAINTEST_PATH)
        

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        Y_test_scaled = scaler_Y.transform(Y_test)

        def rf_cv(n_estimators, max_features, max_depth, max_leaf_nodes):
            rf = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_features=max_features,
                max_depth=int(max_depth),
                max_leaf_nodes=int(max_leaf_nodes),
                random_state=42
            )
            rf.fit(X_train_scaled, Y_train_scaled)
            cv_score = cross_val_score(rf, X_train_scaled, Y_train_scaled, cv=3, scoring='neg_mean_squared_error')
            return cv_score.mean()

        param_bounds = {
            'n_estimators': (5, 150),
            'max_features': (0.1, 0.9),  
            'max_depth': (3, 200),
            'max_leaf_nodes': (3, 200)
        }

        scalers = joblib.load(RNA_SCALER_PATH)
        scaler_X = scalers['scaler_X']
        scaler_Y = scalers['scaler_Y']

        X_train, X_test, Y_train, Y_test = joblib.load(RNA_TRAINTEST_PATH)

        optimizer = BayesianOptimization(
            f=rf_cv,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=50)

        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['max_leaf_nodes'] = int(best_params['max_leaf_nodes'])

        # print(best_params)

        best_rf = RandomForestRegressor(**best_params, random_state=42)
        best_rf.fit(X_train_scaled, Y_train_scaled)

        joblib.dump(best_rf, RNA_MODEL_PATH)

        return best_params

print("SCORE: ", RandomForest.train(RandomForest.tunning_hp()))

RandomForest.make_predict(ha=1.2967064500685144, dec=-60.94322222222222, temp=18, 
                          latitude=dms_to_degrees("-22 32 04"), prev_ha=0.37513611111111106, prev_dec=-28.476599999999998)

print("RESULT", hours_to_hms(1.3385039018773313), degrees_to_dms(-61.17208556757857))
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

"""
utils.py

This file contains the skeleton for the functions you need to implement as part of your homework.
Each function corresponds to a specific task and includes instructions on what is expected.
"""

# Task 2: Data Cleaning
def clean_data(df):
    """
    Task: Data Cleaning
    --------------------
    This function should take a pandas DataFrame as input and return a cleaned DataFrame.
    
    Instructions:
    - Handle missing values in categorical and numerical columns separately.
    - Handle incorrect data points (e.g., negative or null weight values) (I know that there is no weight column!).
    - Ensure that in the cleaned dataframe all the missing or incorrect values are encoded as NaN.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    missing_values_count (dict): A dictionary with the count of missing values per column after cleaning.
    """
    df = df.copy()

    categorical_columns = [
        'Sex', 'ChestPainType', 'FBS', 'RestECG',
        'ExAng', 'Slope', 'Ca', 'Thal'
    ]
    numerical_columns = [
        'Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak'
    ]

    # All ? with NAN first
    df = df.replace('?', np.nan)

    # All are numeric column with values later on category column by its nature
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    valid_values = {
        'Sex': [0, 1],
        'ChestPainType': [1, 2, 3, 4],
        'FBS': [0, 1],
        'RestECG': [0, 1, 2],
        'ExAng': [0, 1],
        'Slope': [1, 2, 3],
        'Ca': [0, 1, 2, 3],
        'Thal': [3, 6, 7],
        'Num': [0, 1, 2, 3, 4]
    }
    column_including_target_as_categorical = categorical_columns + ['Num']
    for col in column_including_target_as_categorical:
        if col in df.columns:
            df.loc[~df[col].isin(valid_values[col]), col] = np.nan

    for col in numerical_columns:
        if col in df.columns:
            if col == 'Oldpeak':
            # Oldpeak can be 0, but not negative
                df.loc[df[col] < 0, col] = np.nan
            elif col == 'Age':
                # Age can't be <18 since adult patients bola hai or > 100
                df.loc[(df[col] < 18) | (df[col] > 100), col] = np.nan
            elif col == 'Chol':
                # Cholesterol should be between 0 and 1000
                df.loc[(df[col] <=0) | (df[col] >= 600), col] = np.nan 
            elif col == 'MaxHR':
                df.loc[(df[col] <=0) | (df[col] >= 220), col] = np.nan  
            else:
                # For other continuous features, 0 or negative are invalid
                df.loc[df[col] <= 0, col] = np.nan

    missing_values_count = df.isna().sum().to_dict()

    return df, missing_values_count


# Task 3: Categorical Features Imputation

def impute_missing_categorical(df_train, df_val, df_test, categorical_columns):
    """
    Task: Categorical Features Imputation
    --------------------------------------
    This function should handle missing values in categorical columns using appropriate techniques.
    We will skip scaling and encoding to keep things simple.
    
    Instructions:
    - Create subsets of the input DataFrames (train, validation, test) with only the categorical columns.
    - Use KNNImputer with k=5 and weights set to 'distance' to fill missing values in the categorical columns.
    - Ensure that the imputed values are approximated to the nearest value in the original dataset for each column, to avoid artifacts like decimal values.
    - If any "new value" is equidistant from two original values, choose the smaller one.
    - Add the column names to the resulting DataFrames after imputation.
    - The imputed dataframes should only contain the categorical columns.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_val (pd.DataFrame): The validation DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    categorical_columns (list): A list of column names corresponding to categorical features.

    Returns:
    pd.DataFrame: The training DataFrame with imputed categorical features.
    pd.DataFrame: The validation DataFrame with imputed categorical features.
    pd.DataFrame: The test DataFrame with imputed categorical features.
    """
    # Making copy of original pd
    X_train_cat = df_train[categorical_columns].copy()
    X_val_cat = df_val[categorical_columns].copy()
    X_test_cat = df_test[categorical_columns].copy()

    # KNN Imputer 
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputer.fit(X_train_cat)

    # Fit using Xtrraind and not transform
    X_train_imputed = imputer.transform(X_train_cat)
    X_val_imputed = imputer.transform(X_val_cat)
    X_test_imputed = imputer.transform(X_test_cat)

    for i, col in enumerate(categorical_columns):
        valid_values = np.sort(df_train[col].dropna().unique())  # original values
        # Function to snap each imputed value to nearest valid value
        def snap_to_nearest(value):
            diffs = np.abs(valid_values - value)
            min_diff = diffs.min()
            nearest_values = valid_values[diffs == min_diff]
            return nearest_values.min()  # pick smaller if tie
        X_train_imputed[:, i] = np.vectorize(snap_to_nearest)(X_train_imputed[:, i])
        X_val_imputed[:, i] = np.vectorize(snap_to_nearest)(X_val_imputed[:, i])
        X_test_imputed[:, i] = np.vectorize(snap_to_nearest)(X_test_imputed[:, i])

    #Convert to Pd dataframe and return 
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=categorical_columns)
    X_val_imputed = pd.DataFrame(X_val_imputed, columns=categorical_columns)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=categorical_columns)

    # returning the value
    return X_train_imputed, X_val_imputed, X_test_imputed


# Task 4: Numerical Features Imputation
def impute_numerical_features(df_train, df_val, df_test, numerical_columns):
    """
    Task: Numerical Features Imputation
    ------------------------------------
    This function should handle missing values in numerical columns using appropriate techniques.
    Again, we will skip scaling to keep things simple.
    
    Instructions:
    - The function should take three datasets as input: df_train, df_val, and df_test.
    - The function should return three datasets as output: train_imputed_lasso, val_imputed_lasso, and test_imputed_lasso.
    - Use LassoRegressor in an iterative fashion to impute missing values in numerical columns.
    - Ensure that the imputation process is consistent and does not use any other imputer.
    - Follow these steps:
        1. Create a subset of the train dataset with only the numerical columns. Call this subset train_num.
        2. Create a subset of the val dataset with only the numerical columns. Call this subset val_num.
        3. Create a subset of the test dataset with only the numerical columns. Call this subset test_num.
        4a. Create a subset of train_num containing the rows with missing values. Call this subset train_num_missing.
        4b. Create a subset of train_num containing the rows without missing values. Call this subset train_num_not_missing.
        5a. Create a subset of val_num containing the rows with missing values. Call this subset val_num_missing.
        5b. Create a subset of val_num containing the rows without missing values. Call this subset val_num_not_missing.
        6a. Create a subset of test_num containing the rows with missing values. Call this subset test_num_missing.
        6b. Create a subset of test_num containing the rows without missing values. Call this subset test_num_not_missing.
        7a. Train a Lasso regression model on the correct subset (I am not telling you which one it is).
        7b. Using a Lasso regression, "predict" the missing values in the subsets that have missing values. Only predict the values in the column with the fewest missing values.
        8. Repeat steps 4-7 until all the missing values are imputed.
        9. Save the results in train_num_imputed_lasso, val_num_imputed_lasso, and test_num_imputed_lasso.
        10. Concatenate the imputed subsets with the subsets that did not contain missing values.
        11. Save the resulting datasets in train_imputed_lasso, val_imputed_lasso, and test_imputed_lasso.
        12. Ensure that the order of the rows in the final datasets matches the order in the original datasets.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_val (pd.DataFrame): The validation DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    numerical_columns (list): A list of column names corresponding to numerical features.

    Returns:
    pd.DataFrame: The training DataFrame with imputed numerical features.
    pd.DataFrame: The validation DataFrame with imputed numerical features.
    pd.DataFrame: The test DataFrame with imputed numerical features.
    """
    # Step1-3
    train_num = df_train[numerical_columns].copy()
    val_num = df_val[numerical_columns].copy()
    test_num = df_test[numerical_columns].copy()

    train_idx, val_idx, test_idx = train_num.index, val_num.index, test_num.index

    train_imputed = train_num.copy()
    val_imputed = val_num.copy()
    test_imputed = test_num.copy()

    while (train_imputed.isnull().any().any() or val_imputed.isnull().any().any() or test_imputed.isnull().any().any()):
        missing_counts = train_imputed.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) == 0:
            break

        col_to_impute = cols_with_missing.sort_values().index[0]
        not_missing = train_imputed[col_to_impute].notnull()

        X_train_fit = train_imputed.loc[not_missing].drop(columns=[col_to_impute])
        y_train_fit = train_imputed.loc[not_missing, col_to_impute]

        # Fill missing predictors temporarily
        X_train_fit = X_train_fit.fillna(X_train_fit.mean())

        if len(X_train_fit) > 0:
            model = Lasso(alpha=0.01, max_iter=2000)
            model.fit(X_train_fit.values, y_train_fit.values)

            for df in [train_imputed, val_imputed, test_imputed]:
                missing_mask = df[col_to_impute].isnull()
                if missing_mask.any():
                    X_missing = df.loc[missing_mask].drop(columns=[col_to_impute]).fillna(X_train_fit.mean())
                    if len(X_missing) > 0:
                        df.loc[missing_mask, col_to_impute] = model.predict(X_missing.values)

    # Fallback mean imputation
    for col in numerical_columns:
        col_mean = train_imputed[col].mean()
        train_imputed[col].fillna(col_mean, inplace=True)
        val_imputed[col].fillna(col_mean, inplace=True)
        test_imputed[col].fillna(col_mean, inplace=True)

    return train_imputed.loc[train_idx], val_imputed.loc[val_idx], test_imputed.loc[test_idx]

def merge_imputed(df_cat, df_num):
    """
    Task: Merge Imputed DataFrames
    -------------------------------
    This function should merge the imputed categorical and numerical DataFrames.
    
    Instructions:
    - Merge the imputed categorical and numerical DataFrames on their indexes.
    - Ensure that the resulting DataFrame contains all columns from both input DataFrames.
    
    Parameters:
    df_cat (pd.DataFrame): The DataFrame with imputed categorical features.
    df_num (pd.DataFrame): The DataFrame with imputed numerical features.

    Returns:
    pd.DataFrame: The merged DataFrame containing both categorical and numerical features.
    """
    merged = pd.concat([df_cat, df_num], axis=1)
    return merged


# Task 5: Classification Using a Single Split
def train_and_evaluate_single_split(X_train, X_val, y_train, y_val, cat_cols,num_cols,model, hp):
    """
    Task: Classification Using a Single Split
    ------------------------------------------
    This function should train a classification pipeline on the training set and evaluate it on the validation set, using the provided parameters.

    Instructions:
    - Create a classification pipeline. It should include:
        - A OneHotEncoder for categorical features (handle_unknown='ignore').
        - A StandardScaler for numerical features.
        - The provided classification model.
    - Use ColumnTransformer to apply the appropriate transformations to categorical and numerical features.
    - Set the model parameters using the provided parameters dictionary.
    - Train the model using the training data (X_train, y_train).
    - Evaluate the model on the validation data (X_val, y_val) using F1 score.
    - Return the evaluation results (F1 score) for the given parameters combination.
    
    Parameters:
    X_train (pd.DataFrame): The training feature set.
    X_val (pd.DataFrame): The validation feature set.
    y_train (pd.Series): The training labels.
    y_val (pd.Series): The validation labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.

    Returns:
    dict: A dictionary containing two keys: 'params' (training parameters) and 'F1 scores' (F1 score). Each key should have the correct value.
    """
    model.set_params(**hp)
    # /Pipeline and Column 
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', model)
    ])

    # fitting values
    pipeline.fit(X_train, y_train.values.ravel())


    y_pred = pipeline.predict(X_val)

    f1 = f1_score(y_val, y_pred)

    return {'params': hp, 'F1 scores': f1}


# Task 6: Classification Using Cross-Validation
def train_and_evaluate_cross_validation(X, y, cat_cols,num_cols,model, hp, cv):
    """
    Task: Classification Using Cross-Validation
    --------------------------------------------
    This function should train and evaluate a classification model using cross-validation.
    
    Instructions:
    - Use cross-validation to train and evaluate the model, with shuffle set to True and using the specified number of folds (cv).
    - For each fold, create a classification pipeline similar to the one in Task 5.
    - Evaluate the model on each fold using F1 score.
    - Return the average F1 score across all folds for each parameter combination.
    - Ensure that the cross-validation process is reproducible (use random_state = 8).
    
    Parameters:
    X (pd.DataFrame): The feature set.
    y (pd.Series): The labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.
    cv (int): The number of cross-validation folds.

    Returns:
    dict: A dictionary containing two keys: 'params' (training parameters) and 'Average F1 scores' (F1 score). Each key should have the correct value.
    """
    f1_scores = []

    # Set model hyperparameters
    model.set_params(**hp)

    
    # ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    # Pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=8)

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]


        y_train_fold = y_train_fold.squeeze()
        y_val_fold = y_val_fold.squeeze()

        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_val_fold)

        f1_scores.append(f1_score(y_val_fold, y_pred))

    return {
        'params': hp,
        'Average F1 scores': np.mean(f1_scores)
    }
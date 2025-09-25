"""
HW1 — Foundations of Data Science
Theme: Data preprocessing, EDA, PCA
Dataset: hotel_bookings.csv

SUBMISSION NOTES
- Implement functions exactly as named.
- Do NOT rename columns, and do NOT drop duplicates (beyond Task 1 rules).
- Expected cleaned shape on the official dataset: (119210, 33).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================ CONSTANTS ============================
# Exact feature list for PCA. Do NOT change order or names.
PCA_FEATURES: List[str] = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_guests",
]

# ============================ TASK 1 ============================
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Task 1 — Load & Light Clean (simple instructions)

    What you will do:
    - Turn the reservation status date into a real date type.
    - Turn obvious numeric columns into real numbers (bad values become NaN).
    - Create a new column total_guests = adults + children + babies.
    - Remove rows where total_guests == 0 (bookings must have guests).
    - If adr (average daily rate) is negative, set it to NaN (do not drop the row).

    Why this matters:
    These steps make sure dates are real dates, numbers are real numbers,
    bookings make sense (no zero guests), and errors are marked as missing values.

    Expected shape on the official dataset:
    - (119210, 33) after cleaning.

    Return:
    - DataFrame with all original columns PLUS 'total_guests'.
    - Do NOT drop duplicates. Do NOT fill other missing values here.
    """
    
    # TODO 1 — Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)


    # TODO 2 — Convert reservation_status_date into a proper datetime (invalid -> NaT)
    # print(df.isnull().sum())
    # print((df['reservation_status_date'].dtype))
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
    
    
    # TODO 3 — Convert obvious numeric columns to numbers (invalid -> NaN)
    numeric_cols = [
        "is_canceled", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "days_in_waiting_list", "adr",
        "required_car_parking_spaces", "total_of_special_requests",
        "arrival_date_year", "arrival_date_day_of_month",
    ]
    
    for column in numeric_cols:
      df[column] = pd.to_numeric(df[column], errors='coerce')


    # TODO 4 — Create total_guests = adults + children + babies (row-wise sum)
      # conflict between todo3 and todo5 otherwise because of na values in children it can never be 11210
    df["children"] = df["children"].fillna(0)
    df['total_guests']= df['adults'] + df['children'] + df['babies']
    
    
    # TODO 5 — Keep only rows where total_guests > 0
    # print("adults",(df['adults']).isna().sum())
    # print((df['children']).isna().sum())
    # print((df['babies']).isna().sum())



    # count how many rows are value 0
    # print(df.shape)
    # ans=0
    # for i in range(len(df)):
    #     if df['total_guests'][i] == 0:
    #         ans+=1
    # print(ans)
    # print("Original:", df.shape)
    # print("Zeros:", (df["total_guests"] == 0).sum())
    # print("NaNs:", df["total_guests"].isna().sum())
    # print("After filter:", df[df["total_guests"] > 0].shape)
    df = df[ df['total_guests'] > 0.0 ]
    # print(df.shape) 


    # TODO 6 — Handle negative prices: if adr < 0, set it to NaN (do NOT drop the row)
    df.loc[df['adr'] < 0, 'adr'] = np.nan


    # df['adr'].isna().sum()
    # TODO 7 — Return the cleaned DataFrame

    return df
   
    
# ============================ TASK 2 ============================
def numeric_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Task 2 — Basic Numeric KPIs (simple instructions)

    What you will do:
    - Take the cleaned DataFrame from Task 1.
    - Compute a few key numbers (no plotting/printing).
    - Return them in a small dictionary.

    KPIs to compute:
    - rows          -> number of rows in the cleaned data
    - cols          -> number of columns
    - cancel_rate   -> mean of is_canceled
    - adr_p95       -> 95th percentile of adr
    - avg_stay_len  -> mean of total nights
                       (stays_in_week_nights + stays_in_weekend_nights)
    """

    # TODO 1 — rows
    rows,cols = df.shape
    # TODO 2 — cols

    # TODO 3 — cancel_rate
    cancel_rate_mean = df['is_canceled'].mean()


    # TODO 4 — adr_p95
    # print(df['adr'].isna().sum())
    # in Task1: last todo is set nan to <0 number now percentile becoming nan 
    # Conflict
    # df2 = df['adr'].dropna()
    # print(df2.shape)
    # dropping the na value to get percentile
    adr_95percentile = np.percentile(df['adr'].dropna(), 95)


    # TODO 5 — avg_stay_len
    avg_stay_len = (df['stays_in_week_nights'] + df['stays_in_weekend_nights']).mean()
    k=  {
            "rows": float(f'{rows:.4f}'),
            "cols": float(f'{cols:.4f}'),
            "cancel_rate": float(f'{cancel_rate_mean:.4f}'),
            "adr_p95": float(f'{adr_95percentile:.4f}'),
            "avg_stay_len": float(f'{avg_stay_len:.4f}')
        }


    # TODO 6 — package & return
    return k

# ============================ TASK 3 ============================
def categorical_cancel_stats(df: pd.DataFrame) -> Dict[str, object]:
    """
    Task 3 — Categorical EDA (simple instructions)

    What you will do:
    - Use the cleaned DataFrame from Task 1.
    - Compute cancellation rates (mean of is_canceled) by:
        * hotel        -> hotel_rates (dict)
        * deposit_type -> deposit_rates (dict)
    - Find the market segment with the highest cancel rate among
      segments with at least 500 rows -> ('segment', rate_float).
    """

    # TODO 1 — hotel_rates
    # print( df['hotel']) 
    # object 
    # print( type(df['hotel'][0])) 
    # string
    # print(df['hotel'].unique())
    # ['Resort Hotel' 'City Hotel']
    grouping_hotel = df.groupby('hotel')['is_canceled'].mean()
    # print(grouping_hotel.to_dict())
    # {'City Hotel': 0.4178593534858457, 'Resort Hotel': 0.27767373336329815} 


    # TODO 2 — deposit_rates
    # print(df['deposit_type'].dtype)
    # print(df['deposit_type'].unique())
    # ['No Deposit' 'Refundable' 'Non Refund']
    grouping_deposit = df.groupby('deposit_type')['is_canceled'].mean()


    # TODO 3 — top_segment_500
    # print(df['market_segment'].dtype)
    # print(df['market_segment'][0]
    # print(df['market_segment'].unique())
    # print(df['market_segment'].isna().sum())   
    # ['Direct' 'Corporate' 'Online TA' 'Offline TA/TO' 'Complementary' 'Groups'
    #  'Undefined' 'Aviation']

    # Top market segment (n ≥ 500) → top_segment_500
    # Count rows per market_segment and keep only categories with at least 500 rows.
    # Filter df to those segments, group by market_segment, compute the mean cancellation rate, and sort descending.
    # Take the first one and return it as a tuple: (segment_name, rate) where rate is a float.
    market_grouping = df.groupby('market_segment').count()
    # print(market_grouping)
    # 500> rows in market segment 
    required_segment = market_grouping[market_grouping['is_canceled'] >= 500]
    # print(required_segment.index.to_list())
    # Filter df to required_segment segments, group by market_segment, compute the mean cancellation rate, and sort descending.
    filter_df = df[df['market_segment'].isin(required_segment.index)]
    mean_cancellation_rate = filter_df.groupby('market_segment')['is_canceled'].mean()
    # print(mean_cancellation_rate)
    sorting_mean_cancellation_rate = mean_cancellation_rate.sort_values(ascending=False)
    # print(sorting_mean_cancellation_rate)
    # print(sorting_mean_cancellation_rate.index[0]) 
    first_row_key_name= sorting_mean_cancellation_rate.index[0]
    # print(first_row_key_name)
    # print(sorting_mean_cancellation_rate[0])  
    # Take the first one and return it as a tuple: (segment_name, rate) where rate is a float.

    # TODO 4 — return dict
    cat = {
    "hotel_rates": grouping_hotel.to_dict(),
    "deposit_rates": grouping_deposit.to_dict(),
    "top_segment_500": (first_row_key_name, float(f'{sorting_mean_cancellation_rate[0]:.4f}'))
    }
    return cat

# ============================ TASK 4 ============================
def build_pca_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Task 4 — Build PCA-ready Matrix (simple instructions)

    What you will do:
    - Use the cleaned DataFrame from Task 1.
    - Select EXACTLY the 14 numeric features in PCA_FEATURES (in order).
    - Median-impute missing values, then standardize (z-score).
    - Return a NumPy array with shape (n_samples, 14), no NaNs.
    """

    # TODO 1 — select columns in order
    # PCA_FEATURES2 = PCA_FEATURES


    # TODO 2 — median imputation
    num_imputer = SimpleImputer(strategy="median")
    # print(df['total_guests'])
    num_data = df[PCA_FEATURES]
    num_fit_transform = num_imputer.fit_transform(num_data)
    # after imputation. <class 'numpy.ndarray'>
    # print(num_fit_transform)  # 2d array
    # check null values in np array
    # print(np.isnan(num_fit_transform)) 
 

    # TODO 3 — standardize
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_fit_transform)
    # print(np.isnan(num_scaled).sum())


    # TODO 4 — return array
    X= num_scaled
    return X

# ============================ TASK 5 ============================
def run_pca(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 5 — Run PCA (simple instructions)

    What you will do:
    - Take the PCA-ready matrix X from Task 4 (no NaNs, standardized; shape: (n_samples, 14)).
    - Fit PCA with n_components=3.
    - Return TWO arrays:
        1) explained_variance_ratio_ -> shape (3,)
        2) components_               -> shape (3, 14)
    """

    # TODO 1 — init PCA
    pca = PCA()


    # TODO 2 — fit
    X_pca = pca.fit_transform(X)

    # TODO 3 — grab arrays
    ratio = pca.explained_variance_ratio_
    comps = pca.components_


    # TODO 4 — return
    ratio, comps = (ratio[0:n_components],comps[0:n_components])
    return ratio,comps
# ============================ END OF FILE ============================
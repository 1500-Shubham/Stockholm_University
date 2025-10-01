# Numpy
np.nan
float(f'{avg_stay_len:.4f}'))
<!-- Calculate in np array -->
np.isnan(array).sum()

# Pandas
df = pd.read_csv("hotel_bookings.csv")
df.head()
pd.concat([num_df,cat_df],axis=1)
df.describe()
df['reservation_status_date'].dtypes
categorical_columns = df_new.select_dtypes(include=['object']).columns.tolist()
df['adr'].count()
df['market_segment'].value_counts() - unique value counts
(df['children']).isna().sum())
negative_adr_count = (df['adr'] < 0).sum()
df.loc[df['adr'] < 0, 'adr'].count()
<!-- Update based on some condition -->
df= df[df['total_guests'] > 0] [true false true for each row value match]
<!-- Filling condition based any column -->
df.loc[df['adr'] < 0, 'adr'] = np.nan
 <!-- if adr < 0, set it to NaN --> df[key] = np.nan
for key,item in enumerate(df['adr']):
    print(key,item)
    df['item]= vcalue then dont actual change value
    need to use .loc
    <!-- OR -->
<!-- return columns to column match mein replace -->
df['children'] = df['children'].fillna(0) 

- See all available methods
    - print(dir(df['reservation_status_date'])) 
    - gave to_datetime then use pd.
- Convert proper datetime and to numbers (invalid -> NaT)
    - df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'],errors='coerce')
    - df[item] = pd.to_numeric(df[item],errors = 'coerse')
<!-- mean for that row .median() . describe()  -->
df['is_canceled'].mean() .quantile(0.95)
<!-- quantile(q)=percentile(100×q) -->
<!-- Grouping and Dict -->
hotel = df.groupby('hotel')['is_canceled'].mean()
hotel.to_dict()
hotel.columns 
for i in range(len(correlation_matrix.columns)) 
<!-- single array after value_count -->
<!-- (8,) -->
column.index can do [0] [1] to get index name
index and column are different things index parameter sets the row labels of a DataFrame.

<!-- sorting a df -->
mean_cancellation_rate.sort_values(ascending=False)
<!-- columns inside value match with a list -->
df[df['market_segment'].isin(required_columns)]

<!-- Df columns select and drop -->
df_numerical = df[PCA_FEATURES] PCA_FEATure = [name1,name2,name3]
df.drop(columns=['column1', 'column2'], inplace=False)

<!-- Correlation Matrix -->
correlation_matrix = df_after_imputation[numerical_columns].corr()

# Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
<!-- Return np array -->
imp = SimpleImputer(strategy="median")
num_imputer = imp.fit_transform(df_numerical)
<!-- Convert numpy array back to pd -->
num_df = pd.DataFrame(a, columns=numerical_columns)

scaler = StandardScaler()
num_scaled = scaler.fit_transform(num_imputer)

pca = PCA(n_components=3)
pca_array = pca.fit(X)
print(pca_array.components_) -> eigen vectors
pca_array.explained_variance_ratio) -> Eigen values hai 

<!-- CREATE XPCA1, XPCA2-->
X_pca= pca.fit_transform(X_scaled)
X_pca[:,0] this gives all rows converted value wrt pca1(eigen vectors)
<!--  Project data --> pca_array.components_ = [PC1,PC2,PC3 so need to tranpose it]
X_pca = X.dot(components.T)
<!-- Whenever trying to get variables good accoring to pca -->
This way we fit the X according to columns
loadings = pd.DataFrame(
    # converting pca values in columns thats why
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(explained_var))],
    index=X.columns
    # index parameter sets the row labels of a DataFrame.
)
loadings["PC1"].sort_values(key=abs, ascending=False)

# Flow For Question
df load , EDA (missing values impute , numerical(median) and categorical(mode) , select features for analysis = Univariate plt histogram, bivariate = x and y both (median se woh feature differ kar raha ache se grouping), multivariate (correlation_matrix))
All API inside on sklearn 

# Numpy
np.random.seed(0)
np.nan
float(f'{avg_stay_len:.4f}'))
round(float(prec_lr), 3)
<!-- Calculate in np array --> Otherwise in df.isna() command hai
np.isnan(array).sum()
<!-- Index capture and create an array -->
mask_array = df_copy['cluster_dbscan'] != -1
print(mask_array)
X_scaled = X_scaled[mask_array]
df_copy = df_copy[mask_array]

# A.shape = (891, 2)
# B.shape = (891, 171)
combined = np.concatenate((A, B), axis=1)

np.corrcoef computes Pearson correlation = covariance / (σX σY).
reason to do x.Transpose this funciton needs a single feature in a single row all values
this need (x1 feautres [all n values], x2 feature [all n values] , y [all n values] )
calculate corre matrix = 3*3 feautres 
[[1.         0.50285852]. toh capture correlation(x,y) using [0,1] index of 2d array
 [0.50285852 1.        ]]

X = np.array([30, 10, 20])
idx = np.argsort(X)
print(idx)         # [1 2 0] now this array can be used to sort X.values column or y_pred

# Pandas
X = pd.concat([X_train_imputed, X_val_imputed], ignore_index=True)
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
<!-- Add a new row on column -->
performance_df.loc[len(performance_df)] = result
Here no need to give column name agar result dictinoary usi form mein
<!-- Update based on some condition -->
df= df[df['total_guests'] > 0] [true false true for each row value match]
<!-- Filling condition based any column -->
df.loc[df['adr'] < 0, 'adr'] = np.nan
df = df.replace('?', np.nan) eveywhere replace
df.loc[~df['Sex'].isin([0, 1]), 'Sex'] = np.nan

y_train = pd.DataFrame([0 if i == 0 else 1 for i in y_train], columns=['Num'])
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
mean_cancellation_rate.sort_values(ascending=False,by='column name')
<!-- columns inside value match with a list -->
df[df['market_segment'].isin(required_columns)]

<!-- Df columns select and drop -->
df_numerical = df[PCA_FEATURES] PCA_FEATure = [name1,name2,name3]
df.drop(columns=['column1', 'column2'], inplace=False)

<!-- Correlation Matrix -->
correlation_matrix = df_after_imputation[numerical_columns].corr()

df_cleaned = df_new.dropna() remove rows with nan vlaues default axis=0

<!-- #One hot encoding of column Gender Male to 0 female to 1 -->
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Gender] = [0 if i=='Male' else 1 for i in df['Gender']]
<!-- If more than 2 distinct variablews : new columns added -->
pd.get_dummies() (recommended for multiple categories)
df = pd.get_dummies(df, columns=['Dummy'], prefix='Dummy')
<!-- Dummy_A   Dummy_B   Dummy_C -->


# Scikit-Learn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
<!-- LAB1 -->
<!-- Return np array -->
imp = SimpleImputer(strategy="median")
num_imputer = imp.fit_transform(df_numerical)
<!-- Convert numpy array back to pd -->
num_df = pd.DataFrame(a, columns=numerical_columns) can also use index

scaler = StandardScaler()
num_scaled = scaler.fit_transform(num_imputer)

pca = PCA(n_components=3)
pca_array = pca.fit(X)
print(pca_array.components_) -> eigen vectors
pca_array.explained_variance_ratio) -> Eigen values hai 

<!-- Printing which column is contributing max to that pca -->
ratio, comps = pca.explained_variance_ratio_, pca.components_
print('variance ratios:', ratio, 'sum:', ratio.sum())
features = hw.PCA_FEATURES
for i, row in enumerate(comps):
    j = np.argmax(np.abs(row)) get index from that complete array row easy
    print(f'PC{i+1} top feature: {features[j]} (loading={row[j]:.3f})')


<!-- CREATE XPCA1, XPCA2-->
X_pca= pca.fit_transform(X_scaled)
X_pca[:,0] this gives all rows converted value wrt pca1(eigen vectors)
<!--  Project data --> pca_array.components_ = [PC1,PC2,PC3 so need to tranpose it]
X_pca = X.dot(components.T)
<!-- Whenever trying to get variables good accoring to pca -->
This way we fit the X according to columns
<!-- Numpy to dataframe with names of columns -->
loadings = pd.DataFrame(
    # converting pca values in columns thats why
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(explained_var))],
    index=X.columns
    # index parameter sets the row labels of a DataFrame.
)
loadings["PC1"].sort_values(key=abs, ascending=False)
x_pca = X.dot(pca_array.components_.T)
print("X1 feauter",X[:,0],"XPCA1",x_pca[:,0])

<!-- LAB2 -->
from sklearn.cluster import KMeans
<!-- Kmeans -->
kmeans = KMeans(n_clusters=k, init='k-means++',random_state=42).fit(X_scaled)
kmeans.labels_ ,interia_ , 
<!-- DBSCAN -->
 dbscan = DBSCAN(eps=e, min_samples=sam).fit(X_scaled)
 dbscan.labels_ This has 1,0,-1 also noise
<!-- Silhoute Score -->
db_sil =silhouette_score(X_scaled, predicted_labels) 
<!-- Cluster Purity Calculate using confusion matrix -->
cm = confusion_matrix(true_labels, predicted_labels)
purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
since majority element jaha belong us cluster ka main point wahi hai


<!-- Lab3 -->
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression

a) r2_hw = lin_hw.score(X_hw, y_hw) x2D and y model useds
r2_score(y_hw, y_pred)   # both (n,) 
<!-- lin.score(X_hw, y_hw)   # internally calls r2_score(y, y_pred) -->
internally prediction call x_hw 2d -> y_pred (n,) then use variance formula ypred and y_hw

b) RMSE X_hw(n,1) -> .ravel() -> flatten (n,) to compare directly
root_mean_squared_error(y_hw, (s * X_hw + b).ravel())


a = df_hw['Height(m)'].values
a = a.reshape(-1,1) 2D -1 = “auto-calculate rows” 1 = “force one column” :
OR df_hw[['Height(m)']].values 2D arrays

 <!-- Model Fit and slope and intercept -->
<!-- # model fit using x[2d] need to reshape(-1,1) and y[1d] -->
lin_hw = LinearRegression().fit(X_hw, y_hw)
lin_hw.coef_ [b1 b2 b3 all features coefficient] -> [0] to get 1st feature 
lin_hw.intercept_
lin_bmi.predict(Xb)

<!-- Creating sample --> min max and number of points
np.linspace(X_hw.min(), X_hw.max(), 200).reshape(-1, 1)

<!-- Pipeline  -->
ridge_pipe = Pipeline([
    ("scaler",StandardScaler()),
    ("ridge",RidgeCV())
    # Pipelines are a convenient way to chain preprocessing steps and the model together, and to be consistent between different models.
    # your pipeline should contain a StandardScaler step and a RidgeCV step, but in what order
])
<!-- Get lamba or alpha value after pipeline worked BEST Fit hone ke baad-->
ridge_alpha = getattr(ridge_pipe.named_steps['ridge'], 'alpha_', None)
OR BELOW 
<!-- Ridge and Lasso Alone -->
<!-- RidgeCV Ridge regression with built-in cross validation. -->
reg = Ridge(alpha=.5) Lasso(alpha=0.1)
reg.fit(X,Y)
reg.coef_
reg.intercept_
RidgeCV().function calling
ridge_pipe.named_steps['ridge'].alpha_ 

<!-- Standard scaler and scaling back using the scaled value -->
scaler = StandardScaler().fit(X_train)
scale_ = scaler.scale_
Back to original
ridge_coefs = ridge.coef_ / scale_


# Flow For Question
- df load , 

- EDA (missing values impute , numerical(median) and categorical(mode) , select features for analysis = Univariate plt histogram, bivariate = x and y both (median se woh feature differ kar raha ache se grouping), multivariate (correlation_matrix))

-  Always Standarize data before giving to any algorithm

- Clustering
    - Standarize PCA
    - Kmeans DBscan algo perform now

- Regression
    - Chossing X and Y correlation dekh lo
    - Different score after predicting

# Hw3 and 4
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score,  precision_score, recall_score, f1_score

X, y = make_classification(n_samples=400, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, class_sep=0.8, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
logit = LogisticRegression()
LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=2000)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
round(float(accuracy_score(y_test, tree.predict(X_test))), 3))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", ))])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
<!-- Name + Transformer or new pipeline + Which columns from X you want to apply that -->
preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
lr = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])

<!-- Getting value back from pipelines example-->
lr.named_steps["model"].coef_.ravel() -> lr.named_steps["model"] return model jo hai now uske function
<!-- Getting value back from column transformer -->
pre = lr.named_steps["preprocess"] -> this is a column transformer (consist of transformer or pipeline in between)
pre.named_transformers_["num"].named_steps["impute"] -> pre.named_transformers_["num"] gives numerical pipeline -> .names_Step return the imputer and its output

<!-- Decision Tree -->
tree = Pipeline([("preprocess", preprocess), ("model", DecisionTreeClassifier(max_depth=4, random_state=42))]).fit(X_train, y_train)
importances = tree.named_steps["model"].feature_importances_ -> for importance of each feature basically gain typw

<!-- Stratified K fold Get indexes and create train and val data : No need to use train test split-->
 skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=8)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        now fit pipeline.fit(X_train_fold,y_train_fold)

<!-- Naive Bayes -->
from sklearn.naive_bayes import MultinomialNB
nb = Pipeline([("preprocess", preprocess), ("model", MultinomialNB(alpha=1.0))]).fit(X_train, y_train)
y_hat_nb = nb.predict(X_test)

<!-- Handling Pipeline -->
numeric = Pipeline([("impute", SimpleImputer(strategy="median")(strategy="most_frequent")//for categorical column), ("scale", MinMaxScaler())])
preprocessing_numeric = numeric.fit(X_train[num_cols])
    <!-- Fit and Tranform like previously we did fit and predict -->
preprocessed_numeric_train = preprocessing_numeric.transform(X_train[num_cols])
<!-- Hamming Distance for categorical Columns Data -->
S-1 # Convert sparse matrices to dense arrays for pairwise_distances
cat_train_dense = preprocessed_categorical_train.toarray()
hamming_dist = pairwise_distances(cat_train_dense, metric='hamming')
euclidean_dist = pairwise_distances(numeric_train, metric='euclidean')
euclidean_dist
net_distance = (2/8)*euclidean_dist + (6/8)*hamming_dist
TEST TRAIN PAIRWISE FOR PREDICTION
euclidean_dist_test = pairwise_distances(numeric_test,numeric_train, metric='euclidean')
hamming_dist_test = pairwise_distances(cat_train_dense, metric='hamming')
net_ditance_test= 
<!-- KNN prediciton using distnaces rather than X_train type -->
Method-1 knn.fit(X_train, y_train) y_pred = knn.predict(X_test)
Method-2 knn.fit(net_distance,y_train) y_pred = knn.predict(net_ditance_test)

<!-- Entering new value in  df with a as json -->
performance_df.loc[len(performance_df)] = a
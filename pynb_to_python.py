


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn import svm
train_df = pd.read_excel('/Users/b.sanjeev/Downloads/snehanshuwork-master/Data_Train.xlsx') test_df=pd.read_excel('/Users/b.sanjeev/Desktop/test2.xlsx')
train_df
big_df = train_df.append(test_df,sort=True) big_df.dtypes
big_df.head(20)
big_df['Total_Stops']=big_df['Total_Stops'].replace('non-stop','0 stop') # In[89]:
big_df['Stop'] = big_df['Total_Stops'].str.split(' ').str[0]
# In[90]:
big_df['Stop'] = big_df['Stop'].astype(int)
# In[91]: big_df=big_df.drop(['Total_Stops'], axis=1) # In[92]:
big_df['Arrival_Hour'] = big_df['Arrival_Time'] .str.split(':').str[0]

 big_df['Arrival_Minute'] = big_df['Arrival_Time'] .str.split(':').str[1]
big_df['Arrival_Hour'] = big_df['Arrival_Hour'].astype(int) big_df['Arrival_Minute'] = big_df['Arrival_Minute'].astype(int) big_df=big_df.drop(['Arrival_Time'], axis=1)
# In[93]:
big_df['Dep_Hour'] = big_df['Dep_Time'] .str.split(':').str[0] big_df['Dep_Minute'] = big_df['Dep_Time'] .str.split(':').str[1] big_df['Dep_Hour'] = big_df['Dep_Hour'].astype(int) big_df['Dep_Minute'] = big_df['Dep_Minute'].astype(int) big_df=big_df.drop(['Dep_Time'], axis=1)
# In[94]:
big_df['Route_1'] = big_df['Route'] .str.split('→ ').str[0] big_df['Route_2'] = big_df['Route'] .str.split('→ ').str[1] big_df['Route_3'] = big_df['Route'] .str.split('→ ').str[2] big_df['Route_4'] = big_df['Route'] .str.split('→ ').str[3] big_df['Route_5'] = big_df['Route'] .str.split('→ ').str[4]
# In[95]:
big_df['Price'].fillna((big_df['Price'].mean()), inplace=True) # In[96]:
big_df['Route_1'].fillna("None",inplace = True) big_df['Route_2'].fillna("None",inplace = True) big_df['Route_3'].fillna("None",inplace = True) big_df['Route_4'].fillna("None",inplace = True) big_df['Route_5'].fillna("None",inplace = True)
# In[97]:
big_df.describe()
# In[98]:
big_df=big_df.drop(['Route'], axis=1) big_df=big_df.drop(['Duration'], axis=1)
from sklearn.preprocessing import LabelEncoder lb_encode = LabelEncoder()


 big_df["Additional_Info"] = lb_encode.fit_transform(big_df["Additional_Info"]) big_df["Airline"] = lb_encode.fit_transform(big_df["Airline"]) big_df["Destination"] = lb_encode.fit_transform(big_df["Destination"]) big_df["Source"] = lb_encode.fit_transform(big_df["Source"]) big_df['Route_1']= lb_encode.fit_transform(big_df["Route_1"]) big_df['Route_2']= lb_encode.fit_transform(big_df["Route_2"]) big_df['Route_3']= lb_encode.fit_transform(big_df["Route_3"]) big_df['Route_4']= lb_encode.fit_transform(big_df["Route_4"]) big_df['Route_5']= lb_encode.fit_transform(big_df["Route_5"])
def missing_values_table(df): # Total missing values mis_val = df.isnull().sum()
# Percentage of missing values
mis_val_percent = 100 * df.isnull().sum() / len(df)
# Make a table with the results
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
# Rename the columns
mis_val_table_ren_columns = mis_val_table.rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'})
# Sort the table by percentage of missing descending mis_val_table_ren_columns = mis_val_table_ren_columns[
mis_val_table_ren_columns.iloc[:,1] != 0].sort_values( '% of Total Values', ascending=True).round(1)
# Print some summary information
print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
"There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
# Return the dataframe with missing information return mis_val_table_ren_columns
# In[101]:
missing_values_table(big_df)
from IPython.display import Image
from IPython.core.display import HTML Image(url = "http://i.imgur.com/QBuDOjs.jpg")
# # LINEAR REGRESSION # In[55]:
#Build our model method lm = LinearRegression()
# In[119]:


 #Build our cross validation method
kfolds = KFold(n_splits=50,shuffle=True, random_state=100)
# In[120]:
def cv_rmse(model):
rmse = np.sqrt(-cross_val_score(model, X, y,
scoring="neg_mean_squared_error", cv = kfolds))
return(rmse)
# In[121]:
benchmark_model = make_pipeline(RobustScaler(),
lm).fit(X=X_train, y=y_train) cv_rmse(benchmark_model).mean()
# # Lasso Regression # In[125]:
alphas = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.01]
alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
lasso_model2 = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,
alphas = alphas2,
random_state = 42)).fit(X_train, y_train)
# In[126]:
scores = lasso_model2.steps[1][1].mse_path_
plt.plot(alphas2, scores, label='Lasso') plt.legend(loc='center') plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.tight_layout() plt.show()
# In[46]: cv_rmse(lasso_model2).mean()


 from mlxtend.regressor import StackingCVRegressor from sklearn.pipeline import make_pipeline
#setup models
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas2,
random_state = 42, cv=kfolds))
xgboost = make_pipeline(RobustScaler(),
XGBRegressor(learning_rate =0.1, n_estimators=200, max_depth=10,
min_child_weight=5 ,gamma=0, subsample=0.7, colsample_bytree=0.8,objective= 'reg:squarederror', nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006))
#stack
stack_gen = StackingCVRegressor(regressors=( lasso,
xgboost), meta_regressor=xgboost,
use_features_in_secondary=True)
from sklearn.metrics import mean_squared_error from math import sqrt
rmse = np.sqrt(mean_squared_error(y_test, stack_gen_preds)) print("RMSE: %f" % (rmse))
# In[78]:
df_test_xgb = df_test[['Additional_Info', 'Airline', 'Destination', 'Source', 'Date', 'Month', 'Year', 'Stop', 'Arrival_Hour', 'Arrival_Minute', 'Dep_Hour',
'Dep_Minute', 'Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']]
preds_1 = stack_gen_model.predict(df_test_xgb) df_test_xgb['Price'] = preds_1 df_test_xgb.to_excel('flight_price69.xlsx')
# In[167]:


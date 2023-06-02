#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


df=pd.read_csv(r"C:\Users\Basu gupts\Downloads\kc_house_data (1).csv")


# In[3]:


df.size


# In[4]:


df.shape


# #Data Preprocessing

# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:



df['date']=pd.to_datetime(df['date'])


# In[8]:


df.dtypes


# In[9]:


df.head()


# In[10]:


df.drop(['id'],axis=1,inplace=True)


# In[11]:


df['year']=df['date'].dt.year


# In[12]:


df['month']=df['date'].dt.month


# In[13]:


df.dtypes


# In[14]:


df.head()


# In[15]:


df.drop(['date'],axis=1,inplace=True)


# In[16]:


df.head()


# In[17]:


df.dtypes

# Univariate Analysis
# In[19]:


df.describe()


# On an average the price of a house in Seattle is 540088.14177
# Average bedrooms:3

# Top 10 price of house in Seattle

# In[20]:


top_10_price=df.sort_values(by='price',ascending=False)[:10]
top_10_price


# In[21]:


plt.figure(figsize=(12,7))
sns.barplot(x='view',y='price',data=top_10_price)
plt.title('Top 10 Price')
plt.show()


# In[24]:


sns.histplot(df['price'])
plt.figure(figsize=(20,20))
plt.show()


# Distribution of Numerical Variables

# In[25]:


df.columns


# In[26]:


features=df[['bedrooms', 'bathrooms', 'sqft_living','floors',
       'waterfront', 'view', 'condition', 'grade','yr_renovated']]


# In[27]:


for feature_col in features:
  pos_data=df[df['price']>=540088.14177][feature_col]
  neg_data=df[df['price']<540088.14177][feature_col]

  plt.figure(figsize=(12,7))


  sns.displot(pos_data,label="Positive",color='green')
  sns.displot(neg_data,label="Negative",color='red')

  plt.legend(loc='upper right')
  plt.title(f"Positive and Negative Distribution Plot For {feature_col}")
  plt.show()


# In[28]:


plt.figure(figsize=(12,8))
sns.distplot(df['price'],hist=True)


# Most of the properties sold are near 1 Million
# 

# B.Living Space Distribution

# In[29]:


plt.figure(figsize=(12,8))
sns.distplot(df['sqft_living'])


# C.Bedrooms and Bathrooms

# In[30]:


plt.figure(figsize=(10,6))
sns.countplot(df['bedrooms'])


# In[31]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)


# In[32]:


plt.figure(figsize=(10,6))
sns.countplot(df['bathrooms'])


# In[33]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bathrooms',y='price',data=df)


# The more bathrooms you have,the more expensive the property becomes

# YEAR_BUILT

# In[34]:


import pylab as P
df['yr_built'].hist(bins=100)
P.show()


# #6.Bivariate Analysis

# 1.Plot Pairplots

# In[35]:


sns.pairplot(data=df)


# Scattergraph of LIving Space and Price

# In[36]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='sqft_living',y='price',data=df)


# 2.	Perform a Chi-square analysis to check whether there is a relationship between
# ●	view and waterfront
# ●	condition and grade
# 

# 1.View and Waterfront

# In[37]:


pip install bioinfokit


# In[38]:


from bioinfokit.analys import stat


# In[39]:


ctab=pd.crosstab(df.waterfront,df.view)
print(ctab)


# In[40]:


res=stat()
res.chisq(df=ctab)
print(res.summary)


# Expected Frequency for view-waterfront

# In[41]:


print(res.expected_df)


# 2.Condition and Grade

# In[42]:


ctab=pd.crosstab(df.condition,df.grade)
print(ctab)


# In[43]:


res=stat()
res.chisq(df=ctab)
print(res.summary)


# In[44]:


print(res.expected_df)


# In[45]:


sns.boxplot(x='waterfront',y='price',data=df)


# plot away from the waterfront view appears to be cheaper as compared to the one with waterfront
# 
# 

# In[46]:


df.corr()['price'].sort_values(ascending=False)


# 3.Pearson Correlation and Heat Map

# In[47]:


correlation=df.corr()


# In[48]:


plt.figure(figsize=(12,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


# #7.Drop any unnecessary columns

# In[49]:


df.head()


# In[50]:


df2=df.drop(['year','month'],axis=1,inplace=True)


# In[51]:


df.head()


# In[52]:


df2=df


# In[53]:


df2.head()


# In[54]:


df2.drop(['zipcode'],axis=1,inplace=True)


# #Remove Outliers

# In[55]:


df2.columns


# In[56]:


col=df2[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
          'grade', 'sqft_above',
       'sqft_basement', 'long',
       'sqft_living15', 'sqft_lot15']]


# In[57]:


sns.boxplot(df2['sqft_lot15'])


# In[58]:


for c in col:
  percentile25=df2[c].quantile(0.25)
  percentile75=df2[c].quantile(0.75)
  iqr=percentile75-percentile25
  upper_limit=percentile75+(1.5*iqr)
  lower_limit=percentile25-(1.5*iqr)
  df2=df2[df2[c]<=upper_limit]
  df2=df2[df2[c]>=lower_limit]
  plt.figure()
  sns.boxplot(y=c, data=df2)


# #Split into train and test set

# In[59]:


df2.columns


# In[60]:


x=df2[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']]


# In[61]:


y=df2['price']


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)


# In[64]:


x_train


# In[65]:


y_train


# #10.Scale the variables

# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


sc=StandardScaler()


# In[68]:


x_train=sc.fit_transform(x_train)


# In[69]:


x_test=sc.transform(x_test)


# #11.MODEL BUILDING

# #1.Linear Regression

# Import Library

# In[70]:


from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[71]:


ln=LinearRegression()


# In[72]:


ln.fit(x_train,y_train)


# In[73]:


#model prediction on trained data
y_pred1=ln.predict(x_train)


# In[74]:


#Model Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('R^2:' ,metrics.r2_score(y_train,y_pred1))
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_train,y_pred1))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_train,y_pred1))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred1))) 


# Vizualize Linear Regression Model(Differences between acutal price and predicted values)

# In[75]:


plt.scatter(y_train,y_pred1)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.title("Prices vs Predicted Prices")
plt.show()


# Predict Test Data

# In[76]:


y_pred_lin=ln.predict(x_test)


# In[77]:


acc_linreg=metrics.r2_score(y_test,y_pred_lin)
print('R^2:',acc_linreg)
print('Adjusterd R^2:',1-(1-metrics.r2_score(y_test,y_pred_lin))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_test,y_pred_lin))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_lin)))


# #2.Random Forest Regressor

# In[78]:


x_train


# In[79]:


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(x_train,y_train)


# In[80]:


y_pred_rf=reg.predict(x_train) 


# In[81]:


y_pred_rf2=reg.predict(x_test)


# In[82]:


#Model evaluation
print('R^2:' ,metrics.r2_score(y_train,y_pred_rf))
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_train,y_pred_rf))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_train,y_pred_rf))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred_rf)))


# Vizualizing Random Forest Regression

# In[83]:


plt.scatter(y_train,y_pred_rf)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.title("Prices vs Predicted Prices")
plt.show()


# In[84]:


#predicting Test Data
y_pred_rf2=reg.predict(x_test)


# In[85]:


#Model Evaluation
acc_rf=metrics.r2_score(y_test,y_pred_rf2)
print('R^2:',acc_rf)
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_test,y_pred_rf2))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_test,y_pred_rf2))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf2)))


# #.SVR (Support Vector Regression)
# #predit discrete vaule 

# In[86]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')


# In[87]:


svr_reg.fit(x_train,y_train)


# In[88]:


#Model Evaluation
y_pred_svr=svr_reg.predict(x_train)


# In[89]:


print('R^2:' ,metrics.r2_score(y_train,y_pred_svr))
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_train,y_pred_svr))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_train,y_pred_svr))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred_svr)))


# In[90]:


plt.scatter(y_train,y_pred_svr)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.title("Prices vs Predicted Prices")
plt.show()


# In[91]:


y_test_svr=svr_reg.predict(x_test)


# In[92]:


acc_svr=metrics.r2_score(y_test,y_test_svr)
print('R^2:',acc_svr)
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_test,y_test_svr))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_test,y_test_svr))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_test_svr)))


# #5.Polynomial Regression Model(Degree=2)

# In[93]:


from sklearn.preprocessing import PolynomialFeatures


# In[94]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[95]:


from sklearn.pipeline import Pipeline


# In[96]:


pipe = Pipeline(Input) 
pipe.fit(x_train, y_train)
print("The predicted values are : " + str(pipe.predict(x_train)))
print("The R^2 score value is : " + str(pipe.score(x_train, y_train)))


# In[97]:


poly=PolynomialFeatures(degree=2,include_bias=False)


# In[98]:


poly_features=poly.fit_transform(x_train)


# In[99]:


poly_features_test=poly.fit_transform(x_test)


# In[100]:


poly_reg_model=LinearRegression()


# In[101]:


poly_reg_model.fit(poly_features,y_train)


# In[102]:


poly_reg_model.fit(poly_features_test,y_test)


# In[103]:


y_test_poly=poly_reg_model.predict(poly_features_test)


# In[104]:


y_pred_poly=poly_reg_model.predict(poly_features)


# In[105]:


plt.figure(figsize=(10,6))
plt.title('Degree 2 Polynomial Regression',size=16)
plt.scatter(y_train,y_pred_poly)
plt.show()


# In[106]:


print('R^2:' ,metrics.r2_score(y_train,y_pred_poly))
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_train,y_pred_poly))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_train,y_pred_poly))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred_poly)))


# Test Evaluation

# In[107]:


acc_poly=metrics.r2_score(y_test,y_test_poly)
print('R^2:',acc_poly)
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_test,y_test_poly))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_test,y_test_poly))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_test_poly)))


# #6.Degree 3 Polynomial Regression model

# In[108]:


poly3=PolynomialFeatures(degree=3,include_bias=False)


# In[109]:


poly_features=poly3.fit_transform(x_train)


# In[110]:


poly_reg_model=LinearRegression()


# In[111]:


poly_reg_model.fit(poly_features,y_train)


# In[112]:


y_pred_poly=poly_reg_model.predict(poly_features)


# In[113]:


plt.figure(figsize=(10,6))
plt.title('Degree 3 Polynomial Regression',size=16)
plt.scatter(y_train,y_pred_poly)
plt.show()


# Model Evaluation

# In[114]:


print('R^2:' ,metrics.r2_score(y_train,y_pred_poly))
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_train,y_pred_poly))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_train,y_pred_poly))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred_poly)))


# In[115]:


#Test Degree3
poly_features=poly.fit_transform(x_test)


# In[116]:


poly_reg_model=LinearRegression()


# In[117]:


poly_reg_model.fit(poly_features,y_test)


# In[118]:


y_test_poly=poly_reg_model.predict(poly_features)


# In[119]:


#Model Evaluation
acc_poly3=metrics.r2_score(y_test,y_test_poly)
print('R^2:',acc_poly3)
print('Adjusterd R^2:' ,1-(1-metrics.r2_score(y_test,y_test_poly))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print('MSE:',metrics.mean_squared_error(y_test,y_test_poly))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_test_poly)))


# #MODELS COMPARED

# In[121]:


models=pd.DataFrame({
    'Model' :['Linear Regression','Random Forest','SVR','Polynomial Regression(D-2)','Polynomial Regression(D-3)'],
    'R-Squared Score':[acc_linreg*100,acc_rf*100,acc_svr*100,acc_poly*100,acc_poly3*100]
})
models.sort_values(by='R-Squared Score',ascending=False)


# #14.Check For Multcollinearity

# In[122]:


import statsmodels.api as sm


# In[123]:


df.columns


# In[124]:


a=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']]


# In[125]:


b=df['price']


# In[126]:


a=sm.add_constant(a)
model=sm.OLS(b,a).fit()


# In[127]:


model.summary()


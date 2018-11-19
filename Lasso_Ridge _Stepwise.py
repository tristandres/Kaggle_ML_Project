
# coding: utf-8

# In[1]:


#Ridge Regression for Post_Train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
Post_train=pd.read_csv('//Users/monazaatari/Desktop/stepwise_forward (4).csv',index_col=0)


# In[2]:


Post_train.head()


# In[3]:


Post_train.columns


# In[3]:


Post_train.info()


# In[4]:


ridge = Ridge()
lasso = Lasso()
x,price = Post_train.drop(['SalePrice'], axis=1),Post_train.SalePrice


# In[13]:


#Split data into train and test
X_train, X_test, y_train, y_test =train_test_split(x, price, test_size=0.2,random_state=0)


# In[14]:


#Lasso with Cross Validation 
lasso = LassoCV(alphas=np.linspace(0.00001,1,100), cv=10)
L=lasso.fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))
lasso.alpha_
lasso.coef_
Error_Tr=(y_train - L.predict(X_train))
ErroT= (y_test - L.predict(X_test))
Tr_rmse = (np.mean(Error_Tr**2))**.5
T_rmse=(np.mean(ErroT**2))**.5
print(Tr_rmse,T_rmse)





# In[8]:


plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('residuals')
plt.show()


# In[9]:


from sklearn.metrics import mean_squared_error
print('MSE train:%.3f,test:%.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))


# In[10]:


from sklearn.metrics import r2_score
r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)
import math
math.sqrt(0.047)


# In[8]:


# Regularized methods (Ridge)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
forward_ridge_model= Ridge(normalize=True,random_state=0)
RMSE = []
coefs  = []
scores=[]
intercepts=[]
alphas=np.linspace(0.001, 5, 1000)
for i in alphas:
    forward_ridge_model.set_params(alpha=i)
    forward_ridge_model.fit(X_train,y_train)
    coefs.append(forward_ridge_model.coef_)
    intercepts.append(forward_ridge_model.intercept_)
    scores.append(forward_ridge_model.score(X_train,y_train))
    train_error = y_train - forward_ridge_model.predict(X_train)
    train_rmse = (np.mean(train_error**2))**.5
    test_error=y_test - forward_ridge_model.predict(X_test)
    test_rmse = (np.mean(test_error**2))**.5
    RMSE.append([train_rmse, test_rmse])
frame = pd.DataFrame(RMSE, index=alphas, columns=['Train RMSE', 'Test RMSE']) 
frame.plot.line()
print(frame['Test RMSE'].min())
frame.loc[frame['Test RMSE'] == frame['Test RMSE'].min()]


# In[9]:


plt.plot(alphas, intercepts)


# In[16]:


print(frame)


# In[10]:


optimal_lambda=frame.loc[frame['Test RMSE'] == frame['Test RMSE'].min()].index.values.astype(float)[0]


# In[8]:


Betas=[]
import numpy as np
#coefficents at optimal lambda with ridge:
Optimum_model=Ridge()
Optimum_model.set_params(normalize=True,random_state=0)
Optimum_model.set_params(alpha=optimal_lambda)
Optimum_model.fit(X_train,y_train)
optimal_s=Optimum_model.score(X_train,y_train)
optimal_stest=Optimum_model.score(X_test,y_test)
Betas.append(Optimum_model.coef_)
print(optimal_s,optimal_stest)


# In[19]:


Betas[0]


# In[43]:


#Lasso regression:
lasso=Lasso()
RMSE_LASSO = []
Coefs  = []
Scores=[]
Intercepts=[]
lasso.set_params(normalize=True,random_state=0)
Alphas=np.linspace(0.0001, 0.01, 1000)
for i in Alphas:
    lasso.set_params(alpha=i)
    lasso.fit(X_train,y_train)
    Coefs.append(lasso.coef_)
    Intercepts.append(lasso.intercept_)
    Scores.append(lasso.score(X_train,y_train))
    Train_error = y_train - lasso.predict(X_train)
    Train_rmse = (np.mean(Train_error**2))**.5
    Test_error=y_test - lasso.predict(X_test)
    Test_rmse = (np.mean(Test_error**2))**.5
    RMSE_LASSO.append([Train_rmse, Test_rmse])
Frame = pd.DataFrame(RMSE_LASSO, index=Alphas, columns=['Train RMSE', 'Test RMSE']) 
print(Frame)
Frame.plot.line()
print(Frame['Test RMSE'].min())
Frame.loc[Frame['Test RMSE'] == Frame['Test RMSE'].min()]



# In[45]:


Optimal_lambda=Frame.loc[Frame['Test RMSE'] == Frame['Test RMSE'].min()].index.values.astype(float)[0]


# In[50]:


Optimum_model1=Lasso()
Optimum_model1.set_params(normalize=True,random_state=0)
Optimum_model.set_params(alpha=Optimal_lambda)
Optimum_model.fit(X_train,y_train)
optimal_s=Optimum_model.score(X_train,y_train)
optimal_stest=Optimum_model.score(X_test,y_test)
Betas.append(Optimum_model.coef_)
print(optimal_s,optimal_stest)
print(OTrain_rmse,OTest_rmse)


# In[38]:


#Random Forest:
from sklearn.ensemble import RandomForestRegressor as rfr
import graphlab as gl
model = gl.random_forest_regression.create(train_data, target='label',
                                           max_iterations=2,
                                           max_depth =  3)
#param_grid = {
   #'max_depth': [103],
   #'max_features': [95],
   #'n_estimators': [290],
}

#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                         #cv = 5, n_jobs = -1, verbose = 2)






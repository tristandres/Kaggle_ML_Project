
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


Post_train=pd.read_csv('//Users/monazaatari/Desktop/stepwise_forward (4).csv',index_col=0)


# In[4]:


Post_train.head()


# In[5]:


names=Post_train.columns
print(names)


# In[6]:


import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
x,price = Post_train.drop(['SalePrice'], axis=1),Post_train.SalePrice

rf = RandomForestRegressor(n_estimators=200,max_features=8,oob_score=True,random_state=0)
X_train, X_test, y_train, y_test =train_test_split(x, price, test_size=0.2,random_state=0)
rf.fit(X_train,y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))
print(rf.get_params())

ErroTe= (y_test - rf.predict(X_test))
Te_rmse = (np.mean(ErroTe**2))**.5
print(Te_rmse)


# In[7]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[10]:


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[11]:


rf_random.fit(X_train, y_train)


# In[12]:


rf_random.best_params_


# In[13]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    return(predictions)
    
evaluate(rf_random,X_test,y_test) 
rf_random.score(X_test,y_test)
#rf_random.score(X_train,y_train)


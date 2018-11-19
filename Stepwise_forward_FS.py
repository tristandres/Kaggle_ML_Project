
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


# read dataset with dummified ordinal and categorical variables
train = pd.read_csv('//Users/monazaatari/Desktop/Kaggle_ML_Project-master 3/Dataset_with_dummies_11_14_3PM.csv',index_col=0)
# read dataset in order to obtain SalesPrice
train1=pd.read_csv('/Users/monazaatari/Downloads/Kaggle_ML_Project-master/Data.csv',index_col=0)


# In[3]:


train.columns


# In[4]:


plt.hist(train1['SalePrice'])


# In[6]:


# get log of salesprice
train['SalePrice']=np.log(train1['SalePrice'])
print(train['SalePrice'])


# In[7]:


explanatory=train.drop(['SalePrice'],axis=1)
X_list=list(explanatory.columns) 
Y=train['SalePrice']


# In[8]:


# testing intercept
from statsmodels.regression.linear_model import OLS
#regr1 = OLS(Y, [1]*Y.shape[0]).fit()
#print(regr1.summary())


# In[9]:


from sklearn.linear_model import LinearRegression
from statsmodels.tools import add_constant
lm = LinearRegression()
#regr = OLS(Y, [1]*Y.shape[0]).fit()
#print(regr.aic)
#print(regr.bic)


# In[10]:


# Using BIC:
# Set initial High BIC
BIC_initial=1470.0133231852574
# c will determine the exit condition from the loop
c=0
# list of features selected BIC method
Features_selected=[]
while c<1:
    BIC=[]
    Feature_Indices=[]
    for j in X_list:
        Model=OLS(Y,add_constant(explanatory[Features_selected+[j]])).fit()
        BIC.append(Model.bic)
        Feature_Indices.append(j)
        print('when adding', j, 'the BIC is', Model.bic)
    Temp = pd.DataFrame({'feature':Feature_Indices,'BIC':BIC})
#    print(Temp.head())
    BestFeature = list(Temp.sort_values(by = 'BIC',ascending=True)['feature'])[0]
    Selected=BIC.index(min(BIC))
    #features_selected.append(X_list[selected])
    BIC_updated=min(BIC)
    print('updated bic',BIC_updated)
    print('initial bic',BIC_initial)
    if BIC_updated>BIC_initial:
        c=1
    else:
        print('new feature added',BestFeature)
        BIC_initial=BIC_updated
        #BIC_track.append(BIC_updated)
        Features_selected.append(BestFeature)
        X_list.remove(BestFeature)
print('final',Features_selected)
        
        



# In[11]:


Final_X1=pd.DataFrame(train[Features_selected])
Final_X1['SalePrice']=train['SalePrice']
print(Final_X1)
#Final_X1.to_csv("Stepwise_Forward.csv", sep=',', encoding='utf-8')
#the model has resulted in 48 selected features


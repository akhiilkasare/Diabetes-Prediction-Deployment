#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('diabetes.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


#creating a correlation matrix

corr_matrix = data.corr()
top_corr_features = corr_matrix.index


# In[9]:


sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.figure(figsize=(20,20))


# In[10]:


true_and_false_values = (data['Outcome'] == 0).value_counts()


# In[11]:


true_and_false_values


# ## Train Test Split

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


data.columns


# In[14]:


x = data.iloc[:,:-1].values
y = data['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)


# ## Finding the missing(zeros) in the dataset

# In[15]:


print('total no. of rows : {0}'.format(len(data)))
print('number of rows missing in Pregnancies: {0}'.format(len(data.loc[data['Pregnancies']==0])))
print('number of rows missing in Glucose: {0}'.format(len(data.loc[data['Glucose']==0])))
print('number of rows missing in BloodPressure: {0}'.format(len(data.loc[data['BloodPressure']==0])))
print('number of rows missing in SkinThickness: {0}'.format(len(data.loc[data['SkinThickness']==0])))
print('number of rows missing in Insulin: {0}'.format(len(data.loc[data['Insulin']==0])))
print('number of rows missing in BMI: {0}'.format(len(data.loc[data['BMI']==0])))
print('number of rows missing in DiabetesPedigreeFunction: {0}'.format(len(data.loc[data['DiabetesPedigreeFunction']==0])))
print('number of rows missing in Age: {0}'.format(len(data.loc[data['Age']==0])))


# ## Using Imputer to replace all the zeros with the mean value
# 

# In[16]:


from sklearn.preprocessing import Imputer


# In[17]:


fill_values = Imputer(missing_values=0, strategy='mean', axis=0)


# In[18]:


x_train = fill_values.fit_transform(x_train)
x_test = fill_values.fit_transform(x_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Applying the Algorithm

# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


random_forest_classifier = RandomForestClassifier()


# In[21]:


## Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV


# In[22]:


# Number of trees in random forest
n_estimators = [120,300,500,800,1200]
# Number of features to consider at every split
max_features = ['log2', 'sqrt',None]
# Maximum number of levels in tree
max_depth = [5,8,15,25,30,None]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[23]:


params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[24]:


random_search = RandomizedSearchCV(estimator=random_forest_classifier, param_distributions = params, 
                                      n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


# In[25]:


random_search.fit(x_train, y_train)


# In[26]:


random_search.best_params_


# In[27]:


# random_forest_classifier = RandomForestClassifier(n_estimators=20,min_samples_split=10,min_samples_leaf=10,
#  max_features='sqrt',max_depth= 15,bootstrap = True)


# In[28]:


# random_forest_classifier.fit(x_train, y_train)


# In[29]:


predict_train_data = random_search.predict(x_test)


# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[31]:


accuracy = accuracy_score(y_test, predict_train_data)


# In[32]:


print("The accuracy score is : {0}".format(accuracy))


# In[33]:


conf_matrix = confusion_matrix(y_test, predict_train_data)


# In[34]:


conf_matrix


# In[35]:


import pickle


# In[36]:


file = open('diabetes_detection.pkl', 'wb')
pickle.dump(random_search, file)


# In[ ]:





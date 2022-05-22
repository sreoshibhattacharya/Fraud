#!/usr/bin/env python
# coding: utf-8

# # Credit Card fraud detection project
# 

# In[1]:


#importing the packages 
import pandas as pd
import numpy as np
import sklearn 
from matplotlib import gridspec


# In[2]:


from scipy.stats import norm
from scipy.stats import multivariate_normal 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#bringing in the dataset
data = pd.read_csv("C:\\Users\\India\\Downloads\\creditcard.csv")


# In[4]:


data.head()


# # 

# In[5]:


#checking for missing values
print("missing values:",data.isnull().values.any())


# In[6]:


#no missing values exist


# In[7]:


#checking the length and breadth of data


# In[8]:


data.shape


# In[9]:


data.describe()


# In[10]:


#count the number of fraud and genuine transacions


# In[11]:


fraud = data[data['Class']==1]
genuine = data[data['Class']==0]
outlier_fraction = len(fraud)/float(len(genuine))
print(outlier_fraction)


# In[12]:


count_fraud = len(data[data['Class']==1])
count_genuine = len(data[data['Class']==0])


# In[13]:


print("Fraud transactions:",count_fraud)
print("Genuine transactions:",count_genuine)


# In[14]:


fraud.Amount.describe() #description of transactions which were fraud


# In[15]:


genuine.Amount.describe()


# # 

# In[16]:


#plotting the corr matrix to check how the features are correlated with each other and depict which features are required 


# In[17]:


corrmatx = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmatx, vmax = 0.8, square=True)
plt.show()


# In[18]:


#in this heatmap we can see that most features are not correlated with each other.V2 and V5 are highly negatively correlated with
#field Amount. Also some correlation between V7 and Amount. 


# In[19]:


X=data.drop(['Class'],axis=1)
Y=data['Class']
print(X.shape)
print(Y.shape)

Xdata = X.values
Ydata = Y.values  #numpy array to get just the values


# In[20]:


#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(Xdata,Ydata,test_size = 0.2, random_state = 42)


# In[21]:


#Building the Random forest classifier 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
YPred = rfc.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix


# In[23]:


n_outliers = len(fraud)
n_error = (YPred != Y_test).sum()
print ("The model is Random Forest Classifier")


# In[24]:


acc = accuracy_score(Y_test,YPred)
print("The accuracy score is {}".format(acc))


# In[25]:


prec_score = precision_score(Y_test,YPred)
print("The precision score is {}".format(prec_score))


# In[26]:


import numpy as np


# In[27]:


rec = recall_score(Y_test,YPred)
print("The recall is {}".format(rec))


# In[28]:


f1_score = f1_score(Y_test,YPred)
print("The f1 score is {}".format(f1_score))


# In[30]:


mathews_corr = matthews_corrcoef(Y_test,YPred)
print("The matthews corrcoef is {}".format(mathews_corr))


# In[34]:


#creating the confusion matrix 
Labels = ['Normal','Fraud']
conf_matrix =confusion_matrix(Y_test,YPred)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix, xticklabels = Labels, yticklabels = Labels, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()


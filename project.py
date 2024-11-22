#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('water_potability.csv')
df


# In[3]:


df.columns = df.columns.str.lower()
df.columns


# In[4]:


for c in df.columns:
    print(c, ': dtype=', df[c].dtype)
    print('unique values:', df[c].nunique())
    print(df[c].unique()[:10])
    print()
    


# In[5]:


df.describe().round(3)


# In[6]:


df.isnull().sum()


# ## Train Test split

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[8]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.potability.values
y_val = df_val.potability.values
y_test = df_test.potability.values

del df_train['potability']
del df_val['potability']
del df_test['potability']


# In[9]:


len(df_train), len(df_test), len(df_val), len(df) == len(df_train)+ len(df_test) +len(df_val)


# # Filling in the missing values

# In[10]:


ph_mean = df_full_train.ph.mean()
sulfate_mean = df.sulfate.mean()
trihalomethanes_mean = df.trihalomethanes.mean()


# In[11]:


df_test.ph = df_test.ph.fillna(ph_mean)
df_val.ph = df_val.ph.fillna(ph_mean)
df_train.ph = df_train.ph.fillna(ph_mean)


df_test.sulfate = df_test.sulfate.fillna(sulfate_mean)
df_val.sulfate = df_val.sulfate.fillna(sulfate_mean)
df_train.sulfate = df_train.sulfate.fillna(sulfate_mean)


df_test.trihalomethanes = df_test.trihalomethanes.fillna(trihalomethanes_mean)
df_val.trihalomethanes = df_val.trihalomethanes.fillna(trihalomethanes_mean)
df_train.trihalomethanes = df_train.trihalomethanes.fillna(trihalomethanes_mean)


# In[12]:


df_train.isnull().sum()


# In[13]:


# Create a figure with subplots (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()
for i, column in enumerate(df.columns):
    if column != 'potability':  # Skip the 'potability' column
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()
plt.show()


# In[14]:


df.potability.value_counts(normalize=True)


# ## Normalize data

# In[15]:


def standardize_data(df_train, df_val, df_test):
    for column in df.columns:
        if column != 'potability':  
            # Calculate the mean and standard deviation for each feature
            mean = df_train[column].mean()
            std = df_train[column].std()

            # Standardize the feature and assign it back to the dataframe
            df_train[column] = (df_train[column] - mean) / std
            df_test[column] = (df_test[column] - mean) / std
            df_val[column] = (df_val[column] - mean) / std

        del df_train['potability'] 
        del df_test['potability'] 
        del df_val['potability'] 

    return df_train, df_val , df_test

df_train, df_val, df_test = standardize_data(df_full_train, df_val ,df_test)


# In[16]:


df_train.head()


# In[17]:


df_train.describe().round(3)


# In[18]:


# Create a figure with subplots (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()

# Plot each feature in the standardized training dataframe
for i, column in enumerate(df_train.columns):
    if column != 'potability':
        sns.histplot(df_train[column].dropna(), bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# ## Mutual information score

# In[19]:


from sklearn.metrics import mutual_info_score
for c in df.columns:
    print(c, mutual_info_score(y_train, df_train[c]).round(4)*100)


# In[ ]:


# we can see that all of our features are important


# # Outlier detection

# In[20]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print(outliers)


# ## Correlation Analysis

# In[21]:


corr = df.corr()
corr.round(2)


# In[22]:


plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f')


# # Model

# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# In[24]:


model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict_proba(df_val)[:,1]
roc_auc_score(y_val, y_pred)


# In[ ]:


classification_report(y_val, y_pred)


# In[ ]:


# Plot the Confusion Matrix
cm = confusion_matrix(y_val, y_pred)


plt.figure(figsize = (12,6))
sn.heatmap(cm, annot=True, fmt='d')
plt.title('Model Confusion Matrix', size=15)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


from sklearn.model_selection import KFold


# In[26]:


kfold = KFold(n_splits=10 , shuffle=True, random_state=1)
next(kfold.split(df_full_train))


# In[27]:


train_idx, val_idx = next(kfold.split(df_full_train))
len(train_idx), len(val_idx), len(df_full_train)


# In[ ]:


from tqdm.auto import tqdm


n_splits = 5
# C is equivalent to the inverse of the regularization parameter
# smaller values specify stronger regularization.

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        X_train = df_train.drop(columns=['potability'])
        X_val = df_val.drop(columns=['potability'])
        

        y_train = df_train.potability.values
        y_val = df_val.potability.values

        model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs').fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]


        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s mean= %.3f - std=%.3f' % (C, np.mean(scores), np.std(scores)))


# In[ ]:


scores


# In[ ]:


df_full_train


# In[ ]:


get_ipython().system(' jupyter nbconvert --to script project.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





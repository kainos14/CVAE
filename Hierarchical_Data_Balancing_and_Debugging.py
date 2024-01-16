#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries that will be needed for the lab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import os, datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model
get_ipython().run_line_magic('load_ext', 'tensorboard')

import random


# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale


import numpy as np
import pandas as pd


# In[4]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[6]:


features = pd.read_csv("C:/Users/GC/Feature_Extract_Fall_Detection/random_extracted_features.csv", index_col = 0)


# In[7]:


features


# In[8]:


features['activity'].value_counts()


# In[17]:


plt.figure(figsize=(15, 8))
features['activity'].value_counts().sort_index().plot(kind = "bar", color='blue', title = "Training Examples by Activity Type")
plt.show()


# In[63]:


from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
features['activity'] = label.fit_transform(features['activity'])
features.head()


# In[64]:


features['activity'].value_counts()


# In[65]:


ADL_NUM = 4600
FALL_NUM = 4600


# In[66]:


D01 = features[features['activity']==0].head(ADL_NUM).copy()
D02 = features[features['activity']==15].head(ADL_NUM).copy()
D03 = features[features['activity']==2].head(ADL_NUM).copy()
D04 = features[features['activity']==4].head(ADL_NUM).copy()
D05 = features[features['activity']==1].head(ADL_NUM).copy()
D06 = features[features['activity']==6].head(ADL_NUM).copy()
D07 = features[features['activity']==7].head(ADL_NUM).copy()
D08 = features[features['activity']==10].head(ADL_NUM).copy()
D09 = features[features['activity']==11].head(ADL_NUM).copy()
D10 = features[features['activity']==16].head(ADL_NUM).copy()
D11 = features[features['activity']==13].head(ADL_NUM).copy()
D12 = features[features['activity']==14].head(ADL_NUM).copy()
D13 = features[features['activity']==8].head(ADL_NUM).copy()
D14 = features[features['activity']==9].head(ADL_NUM).copy()
D15 = features[features['activity']==3].head(ADL_NUM).copy()
D16 = features[features['activity']==12].head(ADL_NUM).copy()
D17 = features[features['activity']==5].head(ADL_NUM).copy()
D18 = features[features['activity']==18].head(ADL_NUM).copy()
D19 = features[features['activity']==17].head(ADL_NUM).copy()


# In[67]:


F01 = features[features['activity']==25].head(FALL_NUM).copy()
F02 = features[features['activity']==32].head(FALL_NUM).copy()
F03 = features[features['activity']==31].head(FALL_NUM).copy()
F04 = features[features['activity']==30].head(FALL_NUM).copy()
F05 = features[features['activity']==29].head(FALL_NUM).copy()
F06 = features[features['activity']==29].head(FALL_NUM).copy()
F07 = features[features['activity']==27].head(FALL_NUM).copy()
F08 = features[features['activity']==26].head(FALL_NUM).copy()
F09 = features[features['activity']==24].head(FALL_NUM).copy()
F10 = features[features['activity']==23].head(FALL_NUM).copy()
F11 = features[features['activity']==22].head(FALL_NUM).copy()
F12 = features[features['activity']==21].head(FALL_NUM).copy()
F13 = features[features['activity']==20].head(FALL_NUM).copy()
F14 = features[features['activity']==19].head(FALL_NUM).copy()
F15 = features[features['activity']==33].head(FALL_NUM).copy()


# In[68]:


df = pd.DataFrame()


# In[69]:


df = pd.concat([df, D01, D02, D03, D04, D05, D06, D07, D08, D09, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19])
df = pd.concat([df, F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12, F13, F14, F15])


# In[70]:


'''
df['acc_pitch'] =  np.arctan(df['ADX_x_acc_mean'] / 
                                    np.sqrt(df['ADX_y_acc_mean'] **2 + df['ADX_z_acc_mean'] **2 ))
df['acc_roll'] =  np.arctan(df['ADX_y_acc_mean'] / 
                                    np.sqrt(df['ADX_x_acc_mean'] **2 + df['ADX_x_acc_mean'] **2 ))
'''


# In[71]:


df = df[['ADX_y_acc_mean','MMA_y_acc_mean','activity']]


# In[72]:


features = df.copy()


# In[73]:


features['activity'].value_counts()


# In[74]:


features = features.replace({'activity':0},0)
features = features.replace({'activity':15},0)
features = features.replace({'activity':2},0)
features = features.replace({'activity':4},0)
features = features.replace({'activity':1},0)
features = features.replace({'activity':6},0)
features = features.replace({'activity':7},0)
features = features.replace({'activity':10},0)
features = features.replace({'activity':11},0)
features = features.replace({'activity':16},0)
features = features.replace({'activity':13},0)
features = features.replace({'activity':14},0)
features = features.replace({'activity':8},0)
features = features.replace({'activity':9},0)
features = features.replace({'activity':3},0)
features = features.replace({'activity':12},0)
features = features.replace({'activity':5},0)
features = features.replace({'activity':17},0)
features = features.replace({'activity':18},0)
features = features.replace({'activity':25},1)
features = features.replace({'activity':32},1)
features = features.replace({'activity':31},1)
features = features.replace({'activity':30},1)
features = features.replace({'activity':29},1)
features = features.replace({'activity':28},1)
features = features.replace({'activity':27},1)
features = features.replace({'activity':26},1)
features = features.replace({'activity':24},1)
features = features.replace({'activity':23},1)
features = features.replace({'activity':22},1)
features = features.replace({'activity':21},1)
features = features.replace({'activity':20},1)
features = features.replace({'activity':19},1)
features = features.replace({'activity':33},1)


# In[75]:


features['activity'].value_counts()


# In[76]:


# 0 -> Normal 1-> Fall


# In[77]:


features.columns


# In[78]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
sns.boxplot(x='activity', y='ADX_y_acc_mean', data=features, showfliers=False, saturation=1)
plt.ylabel('Distribution')

#plt.axhline(y= -800, xmin=0.0, xmax=1,dashes=(5,5), c='g')

plt.xticks(rotation=90)
plt.show()


# In[79]:


adl = features.loc[(features['ADX_y_acc_mean'] < -250)]
fall= features.loc[(features['ADX_y_acc_mean'] >= -250)]


# In[80]:


adl['activity'].value_counts()


# In[81]:


fall['activity'].value_counts()


# In[82]:


Normal_df = adl.loc[(adl['activity'] == 0)]


# In[83]:


Normal_df['activity'].value_counts()


# In[84]:


Normal_df.to_csv("C:/Users/GC/SisFall_Normal.csv")


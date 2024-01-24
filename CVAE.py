# This source is for the SisFall dataset. We only need to modify the related feature names for other data sets.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt

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

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

#Pre-perform Hierarchical Data Balancing procedure
features = pd.read_csv("C:/Users/GC/Extracted_Features_org.csv", index_col=0)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
features['activity']=le.fit_transform(features['activity'])

# Feature Selection

df_corr=df.corr()
matrix=df_corr.to_numpy()
target_row = matrix[-1, :]
indices_above_threshold = np.where(target_row >= 0.4)[0]
column_names = features.columns.tolist()
list0 = features.columns[indices_above_threshold].tolist()
df_total=features[list0]

df_corr=df_total.corr()

plt.figure(figsize=(20,20))
sns.set(font_scale=3)
sns.heatmap(df_corr, annot=True, cbar=False)
plt.show()
sns.set(font_scale=1)

# Denosing
No_noise_df = df_total.copy()
noise_factor = 0.5
df['ADX_y_acc_mean'] = df['ADX_y_acc_mean'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=num)
df['ADX_y_acc_max'] = df['ADX_y_acc_max'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=num)
df['MMA_y_acc_mean'] = df['MMA_y_acc_mean'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=num)
df['MMA_y_acc_max'] = df['MMA_y_acc_max'] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=num)

# Noisy Data
X_train, X_test = train_test_split(df_total, test_size=0.1, random_state=42)
X_train.shape, X_test.shape

# Validation data without noisy
X_train2, X_test2 = train_test_split(No_noise_df, test_size=0.1, random_state=42)
X_train2.shape, X_test2.shape

# Normal = 0 / Fall = 1
normal = X_train[X_train['activity'] == 0]
normal.shape
normal2 = X_train2[X_train2['activity'] == 0]
normal.shape

# Data Debugging
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.scatterplot(data=normal2, x='ADX_y_acc_mean', y='MMA_y_acc_mean', hue='activity')
plt.title('Seaborn Scatter Plot with Labels')
plt.show()

#Pre-calculating threadhold value by Exploratory Data Analysis (EDA)
normal = normal[(normal['ADX_y_acc_mean'] > -250)]
normal2 = normal2[(normal2['ADX_y_acc_mean'] > -250)]

y_train = normal['activity']
X_train_normal_train = normal.drop(['activity'], axis=1)
y_train2 = normal2['activity']
X_train_normal_train2 = normal2.drop(['activity'], axis=1)
y_test = X_test['activity']
X_test = X_test.drop(['activity'], axis=1)
y_test2 = X_test2['activity']
X_test2 = X_test2.drop(['activity'], axis=1)
X_train_ft = X_train_normal_train.values
X_train_ft2 = X_train_normal_train2.values
X_test = X_test.values
X_test2 = X_test2.values

X_train_ft.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler = StandardScaler()
scaler.fit(X_train_ft)
scaler.fit(X_test)

X_train_ft2 = scaler.transform(X_train_ft)
X_train_ft = scaler.transform(X_train_ft2)
X_test = scaler.transform(X_test)
X_test2 = scaler.transform(X_test2)

timestamp = 20

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

y_train_array = np.array(y_train)
train_set = np.c_[X_train_ft, y_train_array]

y_train_array = np.array(y_train)
train_set2 = np.c_[X_train_ft2, y_train_array]

y_test_array = np.array(y_test)
test_set = np.c_[X_test, y_test_array]

y_test_array = np.array(y_test2)
test_set2 = np.c_[X_test2, y_test_array]

X_train, y_train = split_sequences(train_set, timestamp)
print(X_train.shape, y_train.shape)
X_train2, y_train = split_sequences(train_set2, timestamp)
print(X_train.shape, y_train.shape)
X_test, y_test = split_sequences(test_set, timestamp)
print(X_test.shape, y_test.shape)
X_test2, y_test2 = split_sequences(test_set2, timestamp)
print(X_test2.shape, y_test2.shape)

X_train.shape, X_train2.shape, X_test.shape, y_train.shape, y_test.shape

import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MultiHeadAttention
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Masking, TimeDistributed, Lambda
from tensorflow.keras.losses import mse, binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras_self_attention import SeqWeightedAttention, SeqSelfAttention
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense, Conv1D, Conv1DTranspose
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import seaborn as sns
import matplotlib.pyplot as plt

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

latent_dim = 2
inter_dim = 3
timesteps, features = X_train.shape[1], X_train.shape[2]

def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0] 
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + z_log_sigma * epsilon

def vae_loss(inputs, decoded):
   
    xent_loss = K.sum(K.binary_crossentropy(inputs, decoded), axis=1)
    xent_loss = 10 * K.mean(metrics.mean_squared_error(x_, x_decoded_mean_), axis=-1)
    return K.mean(xent_loss + kl_loss)

# timesteps, features

input_x = Input(shape=(timesteps, features), name='InputTimeSeries') 

#intermediate dimension 
h = Conv1D(filters=48, kernel_size=5, padding="same", strides=1, activation="relu")(input_x)
h = Conv1D(filters=32, kernel_size=5, padding="same", strides=1)(h)
h = Conv1D(filters=16, kernel_size=5, padding="same", strides=1)(h)
h = LSTM(inter_dim, activation='relu', return_sequences=True)(h)
h = SeqSelfAttention(attention_activation='sigmoid')(h)
h = LSTM(inter_dim, activation='relu', return_sequences=False)(h)
h = Dense(inter_dim, activation='relu')(h)
h = Dense(inter_dim, activation='relu')(h)

#z_layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_sigma])

# Reconstruction decoder
decoder1 = RepeatVector(timesteps)(z)
decoder1 = Dense(inter_dim, activation='relu')(decoder1)
decoder1 = Dense(inter_dim, activation='relu')(decoder1)
decoder1 = LSTM(inter_dim, activation='relu', return_sequences=True)(decoder1)
decoder1 = LSTM(inter_dim, activation='relu', return_sequences=True)(decoder1)
decoder1 = SeqSelfAttention(attention_activation='sigmoid')(decoder1)
decoder1 = Conv1DTranspose(filters=16, kernel_size=5, padding="same", strides=1, activation="relu")(decoder1)
decoder1 = Conv1DTranspose(filters=32, kernel_size=5, padding="same", strides=1, activation="relu")(decoder1)
decoder1 = Conv1DTranspose(filters=48, kernel_size=5, padding="same", strides=1, activation="relu")(decoder1)
decoder1 = TimeDistributed(Dense(features))(decoder1)

model = Model(input_x, decoder1)
model.compile(optimizer='adam', loss=vae_loss, metrics=['accuracy'])
model.summary()

history = model.fit(X_train, X_train,
                        shuffle=True,
                        epochs=100,
                        validation_data =(X_train2, X_train2),                        
                        batch_size=256)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracys')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

from sklearn.metrics import mean_squared_error

valid_x_predictions = model.predict(X_test)
error = flatten(X_test) - flatten(valid_x_predictions)

valid_mse = np.mean(np.power(flatten(X_test) - flatten(valid_x_predictions), 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': valid_mse,'true_class': y_test})
error_df.describe()
error_df['true_class'].value_counts()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
sns.boxplot(x='true_class', y='reconstruction_error', data=error_df, showfliers=False, saturation=1)
plt.ylabel('Distribution')
plt.axhline(y= 0.1, xmin=0.01, xmax=1,dashes=(5,5), c='g')
plt.xticks(rotation=90)
plt.show()

threshold =  0.10

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='^', ms=3.5, linestyle='',
            label= "Normal" if name == 0 else "Fall")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

LABELS = ["Normal", "Fall"]

y_pred = [0 if e > threshold else 1 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

precision, recall, f1,_ = precision_recall_fscore_support(y_test,y_pred,average='binary')
print ('Accuracy Score :',accuracy_score(error_df.true_class, y_pred) )
print ('Precision :',precision )
print ('Recall :',recall )
print ('F1 :',f1 )

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='(AUC = {:.2f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

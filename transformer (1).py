


tappy_data_path='/home1/vulli/physionet.org/files/tappy/1.0.0/data/Tappy Data'
users_data_path= '/home1/vulli/physionet.org/files/tappy/1.0.0/users/Archived users'

index = 9
chunk_size=100
sliding_step=50

import re
import glob
import os
import random
import sys
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

nb_data_per_person = np.array([0])
def get_user_data():
    import os
    directory = users_data_path
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename)) as f:
                file_contents = f.read()
            file_data = {}
            for line in file_contents.split('\n'):
                if line.strip() != '':
                    key, value = line.split(': ')
                    file_data[key.strip()] = value.strip()

            file_data['ID'] = filename.split("_")[1].split(".")[0]
            # Append the file data to the list
            data.append(file_data)
    df = pd.DataFrame(data)
    df = df[["ID", "Parkinsons"]]
    df["Parkinsons"] = df["Parkinsons"].replace({"False": 0, "True": 1})
    parkinsons_dict = df.set_index('ID')['Parkinsons'].to_dict()
    return parkinsons_dict

positive_files = []
negative_files = []
fils = sorted(glob.glob(os.path.join(tappy_data_path, '*txt')))
random.shuffle(fils)
parkinsons_dict = get_user_data()
files = []
for file in fils:
    f = file.split("/")[index].split("_")[0]
    if f in parkinsons_dict:
        files.append(file)
        if parkinsons_dict[f] == 1:
          positive_files.append(file)
        else:
          negative_files.append(file)
print(len(files), len(positive_files), len(negative_files))

from sklearn.model_selection import train_test_split
from datetime import datetime
train, test = train_test_split(files,  train_size=0.8,  shuffle=True)

def gen_data2(temp, had_parkinson_user, data_list, input, output):
    count = 0
    # print("gen", temp.shape)
    
    nb_datas = int(temp.shape[0] - chunk_size)
    num_cols = temp.shape[1]-1

    for start in range(0, nb_datas, sliding_step):
        end = start + chunk_size
        data = temp[start:end, :]
        data[:,0] = (data[:,0] - data[:,0].mean())/data[:,0].std()
        data[:,1] = (data[:,1] - data[:,1].mean())/data[:,1].std()
        data[:,2] = (data[:,2] - data[:,2].mean())/data[:,2].std()
        # print("differenece",temp[end-1, num_cols] - temp[start, num_cols])
        time_list.append(temp[end-1, num_cols] - temp[start, num_cols])
        if  temp[end-1, num_cols] - temp[start, num_cols] <1000:
          # print("gen", data[:,:-1].shape)
          input.append(data[:,:-1])
          # cv+=1
          output.append(had_parkinson_user)
          count = count + 1
        
        


def one_hot(df, value, cat):
    one_hot_df = pd.get_dummies(df[value], prefix=value, columns=cat)
    df = pd.concat([df, one_hot_df], axis=1)
    df.drop(value, axis=1, inplace=True)
    return df

time_list = []
cv=0
train_inputs = []
train_labels = []
test_inputs = []
test_labels = []
def get_system_time(row):
    # Combine the date and time columns into a single datetime object
    dt = datetime.combine(row['date'], row['time'])
    # Calculate the system time in seconds
    return (dt - datetime(1970, 1, 1)).total_seconds()
def load_data(files, input, output, data_type = 1):
  
  for i in range(0, len(files)):
      # print(files[i])
      print("i:", i)
      
      try:
        ref_date = pd.Timestamp('2000-01-01')
        data = pd.read_csv(files[i], sep = "\t").iloc[:, :-1]
        data.columns = ["user_id","date","time","hand", "hold_time", "dir", "latency_time", "flight_time"]

        
        data = data[np.in1d(data['hand'], ["L", "R", "S"])
                    & data['date'].apply(lambda x: re.search(r'^\d{6}$', str(x)) is not None)
                    & data['time'].apply(lambda x: re.search(r'^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.\d{3}$', str(x)) is not None)
                    & data['hold_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                    & np.in1d(data['dir'], ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS'])
                    & data['latency_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                    & data['flight_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)]

        # print(data)
      
        data['date'] = (pd.to_datetime(data['date'], format='%y%m%d') )
        data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S.%f').dt.time
        # data['time'] = data['time'].apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)).dt.total_seconds() / 86400
        data['datetime'] = data.apply(get_system_time, axis=1)
        # data['datetime'] = data['date'] + " "+ data['time']
        # print(data['date'][0], data['time'][0], data['datetime'][0])
        # data['datetime'] = data['date'] + data['time']/(24*60*60)
        had_parkinson_user = parkinsons_dict[data.iloc[:, 0].unique()[0]]
        data.drop(data.columns[[0, 1, 2]], axis=1, inplace=True) 
        categories_dir = ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS']
        categories_hand = ['L','R','S']
        data = one_hot(data, "hand", categories_hand)
        data = one_hot(data, "dir", categories_dir)
        data = data.reindex(columns=['hold_time', 'latency_time', 'flight_time', 'hand_L', 'hand_R',
          'hand_S', 'dir_LL', 'dir_LR', 'dir_LS', 'dir_RL', 'dir_RR', 'dir_RS',
          'dir_SL', 'dir_SR', 'dir_SS', 'datetime'], fill_value = 0)
      except Exception as e:
        print(e)
        continue
      # if data_type == 1 and had_parkinson_user == 0 and data.shape[0] < 14000:
      #   print("Before", data.shape)
      #   data = apply_smote(data)
      #   print("After", data.shape)
      # print(data.columns)
      data = data.to_numpy().astype('float64')
      if data.shape[0] < chunk_size:
        print("here")
        print(data[:,:-1].shape)
        input.append(data[:,:-1])
        
        output.append(had_parkinson_user)
        continue
      gen_data2(data, had_parkinson_user, nb_data_per_person, input, output)
      # print(input, output)

load_data(train, train_inputs, train_labels)




load_data(test, test_inputs, test_labels, data_type = 0)






train_inputs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
    train_inputs, padding="post"
))
train_labels = np.array(train_labels)

test_inputs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
    test_inputs, padding="post"
))

test_labels = np.array(test_labels)

print(train_inputs.shape)
print(train_labels.shape)
print(test_inputs.shape)
print(test_labels.shape)



import re
import glob
import os
import random
import sys
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

import numpy as np
np.random.seed(2)
import tensorflow as tf
tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Conv1D

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import backend as K
import uuid

def add_pos_2(input,nb):
    input_pos_encoding = tf.constant(nb, shape=[input.shape[1]], dtype="int32")/input.shape[1]
    input_pos_encoding = tf.cast(tf.reshape(input_pos_encoding, [1,10]),tf.float32)
    input = tf.add(input ,input_pos_encoding)
    return input

def stack_block_transformer(num_transformer_blocks):
    input1 = keras.Input(shape=(100, 1))
    x = input1
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x,100,2)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(10, activation='selu')(x)
    return input1,x

def stack_block_transformer_spatial(num_transformer_blocks,x):
  for _ in range(num_transformer_blocks):
      x = transformer_encoder(x,10*15,2)
  x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

  return x

def transformer_encoder(inputs,key_dim,num_heads):
    dropout=0.1
    # Normalization and Attention
    print("transformer_encoder",inputs.shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(key_dim, activation='softmax')(x)
    return x + res


def multiple_transformer(nb):
    '''

    :param nb: number of features ( indicates the number of parallel branches)
    :return:
    '''
    # initialise with the first input

    num_transformer_blocks = 2  #hyperparameter
    input_, transformer_ = stack_block_transformer(num_transformer_blocks)
    transformers = []
    inputs = []
    transformers.append(transformer_)
    inputs.append(input_)
    for i in range(1,nb ):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i) 
        transformer_i = add_pos_2(transformer_i,i)
        transformers.append(transformer_i)
  
    x = layers.concatenate(transformers, axis=-1)
    x = tf.expand_dims(x, -1) #-1 denotes the last dimension
    x = stack_block_transformer_spatial(num_transformer_blocks,x)
    x = Dropout(0.1)(x)
    x = layers.Dense(100, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.1)(x)
    answer = layers.Dense(1, activation='sigmoid')(x)
  
    model = Model(inputs, answer)
    opt = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'],experimental_run_tf_function=False)
    print(model.summary())
    return model

print(train_inputs.shape)
print(train_labels.shape)
print(test_inputs.shape)
print(test_labels.shape)

X_train = train_inputs
y_train = train_labels
X_val = test_inputs
y_val = test_labels

lr = 0.001
model = multiple_transformer(X_train.shape[2])
for i in (np.arange(1,4)*5):  # 10-20    1-10
    print("Y_train")
    print(len(y_train))
    history = model.fit(np.split(X_train,X_train.shape[2], axis=2), \
                        # history  = model.fit(X_data,\
                        y_train, \
                        verbose=1, \
                        shuffle=True, \
                        epochs= 5,\
                        batch_size=110, \
                        validation_data=(np.split(X_val, X_val.shape[2], axis=2), y_val), \
                        )
    lr =  lr / 2
    rms = optimizers.Nadam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

# lr = 0.001
# model = multiple_transformer(data.X.shape[2])

# data.separate_fold(1)

# model = train(model, data, lr)
# print('Validation !!')

y_pred=model.predict(np.split(X_val, X_val.shape[2], axis=2))

# Se =
# T P
# T P + F N
# (1)
# Sp =
# T N
# T N + F P
# (2)
# Acc =
# T P + T N
# T P + T N + F P + F N


labels = np.where(y_pred >= 0.5, 1, 0)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, labels)
print(cm)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
snsitivity = tp / (tp + fn)
acc = (tp + tn)/(tp + tn + fp + fn)


print("specificity:", specificity)
print("snsitivity:", snsitivity)
print("acc:", acc)


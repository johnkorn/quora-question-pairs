########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import pickle
import sys


########################################
## set directories and parameters
########################################
BASE_DIR = 'DATA/'
#EMBEDDING_FILE = '/media/johnkorn/New Volume/DATA/EMBEDDINGS/GoogleNews-vectors-negative300.bin'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'

TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = 256 #np.random.randint(175, 275)
num_dense = 256 #np.random.randint(100, 150)
rate_drop_lstm = 0.4 #0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.4 #0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)


########################################
## LOAD THE DATA
########################################

data_1 = pickle.load(open('DATA/processed/traindata1.pkl', 'rb'))
data_2 = pickle.load(open('DATA/processed/traindata2.pkl', 'rb'))
labels = pickle.load(open('DATA/processed/trainlabels.pkl', 'rb'))

embedding_matrix = pickle.load(open('DATA/processed/w2vmatrix.pkl', 'rb'))
nb_words = embedding_matrix.shape(0)


test_data_1 = pickle.load(open('DATA/processed/testdata1.pkl', 'rb'))
test_data_2 = pickle.load(open('DATA/processed/testdata2.pkl', 'rb'))
test_ids = pickle.load(open('DATA/processed/testids.pkl', 'rb'))

########################################
## sample train/validation data
########################################
np.random.seed(123)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344


########################################
## define the model structure
########################################

embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None


########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
model.summary()
print('Training model {} (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)'.format(STAMP))
logs_path = './lstm_logs/{}'.format(STAMP)
#!mkdir logs_path

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
tensor_board = TensorBoard(log_dir=logs_path)


hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=512, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, tensor_board])


model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])


########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=1024, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=1024, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
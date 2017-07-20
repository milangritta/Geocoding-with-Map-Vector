# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Input, concatenate
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten
from preprocessing import generate_arrays_from_file, GRID_SIZE
from subprocess import check_output

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))

UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 100, 200
print(u"Dimension:", dimension)
print(u"Input length:", input_length)

vocabulary = cPickle.load(open(u"data/vocabulary.pkl"))
print(u"Vocabulary Size:", len(vocabulary))
#  --------------------------------------------------------------------------------------------------------------------
print(u'Preparing vectors...')
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])

vectors = {UNKNOWN: np.ones(dimension), PADDING: np.ones(dimension)}
for line in codecs.open(u"../data/glove.twitter." + str(dimension) + "d.txt", encoding="utf-8"):
    if line.strip() == u"":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]
print(u'Loaded Twitter vectors...', len(vectors))

for line in codecs.open(u"../data/glove." + str(dimension) + "d.txt", encoding="utf-8"):
    if line.strip() == u"":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]
print(u'Loaded GloVe vectors...', len(vectors))

weights = np.zeros((len(vocabulary), dimension))
for w in vocabulary:
    if w in vectors:
        weights[word_to_index[w]] = vectors[w]
weights = np.array([weights])
print(u'Done preparing vectors...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
left_pair = Input(shape=(input_length,))
lp = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_pair)
lp = Conv1D(500, 2, activation='relu', strides=1)(lp)
lp = GlobalMaxPooling1D()(lp)
lp = Dense(100)(lp)
lp = Dropout(0.5)(lp)

left_single = Input(shape=(input_length,))
ls = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_single)
ls = Conv1D(500, 1, activation='relu', strides=1)(ls)
ls = GlobalMaxPooling1D()(ls)
ls = Dense(100)(ls)
ls = Dropout(0.5)(ls)

right_pair = Input(shape=(input_length,))
rp = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(right_pair)
rp = Conv1D(500, 2, activation='relu', strides=1)(rp)
rp = GlobalMaxPooling1D()(rp)
rp = Dense(100)(rp)
rp = Dropout(0.5)(rp)

right_single = Input(shape=(input_length,))
rs = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(right_single)
rs = Conv1D(500, 1, activation='relu', strides=1)(rs)
rs = GlobalMaxPooling1D()(rs)
rs = Dense(100)(rs)
rs = Dropout(0.5)(rs)

entities = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
ent = Conv2D(1, (2, 2), strides=(2, 2), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(entities)
# ent = MaxPooling2D(data_format="channels_first")(ent)
ent = Flatten()(ent)
ent = Dropout(0.5)(ent)

target = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
tar = Conv2D(1, (1, 1), strides=(1, 1), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(target)
# tar = MaxPooling2D(data_format="channels_first")(tar)
tar = Flatten()(tar)
tar = Dropout(0.5)(tar)

merged = concatenate([lp, ls, rp, rs, ent, tar])
merged = Dense(units=(180 / GRID_SIZE) * (360 / GRID_SIZE), activation='softmax')(merged)
model = Model(inputs=[left_pair, left_single, right_pair, right_single, entities, target], outputs=[merged])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights_all_cnn", verbose=0)
# checkpoint = ModelCheckpoint(filepath="../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
# early_stop = EarlyStopping(monitor='acc', patience=5)
file_name = u"data/eval_wiki.txt"
model.fit_generator(generate_arrays_from_file(file_name, word_to_index, input_length, oneDim=False),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / 64,
                    epochs=100, callbacks=[checkpoint])

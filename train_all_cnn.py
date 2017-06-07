# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Input, merge, Model
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten
from preprocessing import generate_arrays_from_file, GRID_SIZE
from subprocess import check_output

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
left_pair = Input(shape=(input_length,), dtype='int32')
lp = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_pair)
lp = Conv1D(500, 2, activation='relu', subsample_length=1)(lp)
lp = GlobalMaxPooling1D()(lp)
lp = Dense(100)(lp)
lp = Dropout(0.5)(lp)

left_single = Input(shape=(input_length,), dtype='int32')
ls = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_single)
ls = Conv1D(500, 1, activation='relu', subsample_length=1)(ls)
ls = GlobalMaxPooling1D()(ls)
ls = Dense(100)(ls)
ls = Dropout(0.5)(ls)

right_pair = Input(shape=(input_length,), dtype='int32')
rp = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(right_pair)
rp = Conv1D(500, 2, activation='relu', subsample_length=1)(rp)
rp = GlobalMaxPooling1D()(rp)
rp = Dense(100)(rp)
rp = Dropout(0.5)(rp)

right_single = Input(shape=(input_length,), dtype='int32')
rs = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(right_single)
rs = Conv1D(500, 1, activation='relu', subsample_length=1)(rs)
rs = GlobalMaxPooling1D()(rs)
rs = Dense(100)(rs)
rs = Dropout(0.5)(rs)

entities = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), dtype='float32')
ent = Conv2D(50, 3, 3, activation='relu', subsample=(3, 3), dim_ordering='th',
                  input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(entities)
ent = MaxPooling2D(dim_ordering='th')(ent)
ent = Flatten()(ent)
ent = Dropout(0.5)(ent)

target = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), dtype='float32')
tar = Conv2D(50, 3, 3, activation='relu', subsample=(3, 3), dim_ordering='th',
                input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(target)
tar = MaxPooling2D(dim_ordering='th')(tar)
tar = Flatten()(tar)
tar = Dropout(0.5)(tar)

merged = merge([lp, ls, rp, rs, ent, tar], mode='concat', concat_axis=1)
lat = Dense(output_dim=180 / GRID_SIZE, activation='softmax', input_dtype='float32')(merged)
lon = Dense(output_dim=360 / GRID_SIZE, activation='softmax', input_dtype='float32')(merged)
model = Model(input=[left_pair, left_single, right_pair, right_single, entities, target], output=[lat, lon])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights_all_cnn", verbose=0)
# checkpoint = ModelCheckpoint(filepath="../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
# early_stop = EarlyStopping(monitor='acc', patience=5)
file_name = u"data/eval_wiki.txt"
model.fit_generator(generate_arrays_from_file(file_name, word_to_index, input_length, oneDim=False),
                     samples_per_epoch=int(check_output(["wc", file_name]).split()[0]),
                     nb_epoch=100, callbacks=[checkpoint])

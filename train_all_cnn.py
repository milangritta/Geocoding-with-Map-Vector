# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Merge, Input
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
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
left_pair = Input(shape=(input_length,), dtype='int32', name='left_pair')
left_pair = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_pair)
left_pair = Conv1D(500, 2, activation='relu', subsample_length=1)(left_pair)
left_pair = GlobalMaxPooling1D()(left_pair)
left_pair = Dense(100)(left_pair)
left_pair = Dropout(0.5)(left_pair)

left_single = Input(shape=(input_length,), dtype='int32', name='left_single')
left_single = Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights)(left_single)
left_single = Conv1D(500, 1, activation='relu', subsample_length=1)(left_single)
left_single = GlobalMaxPooling1D()(left_single)
left_single = Dense(100)(left_single)
left_single = Dropout(0.5)(left_single)

model_right_pair = Sequential()
model_right_pair.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_right_pair.add(Conv1D(500, 2, activation='relu', subsample_length=1))
model_right_pair.add(GlobalMaxPooling1D())
model_right_pair.add(Dense(100))
model_right_pair.add(Dropout(0.5))

model_right_single = Sequential()
model_right_single.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_right_single.add(Conv1D(500, 1, activation='relu', subsample_length=1))
model_right_single.add(GlobalMaxPooling1D())
model_right_single.add(Dense(100))
model_right_single.add(Dropout(0.5))

model_entities = Sequential()
model_entities.add(Conv2D(50, 3, 3, activation='relu', subsample=(3, 3), dim_ordering='th', input_shape=(1, (180 / GRID_SIZE), (360 / GRID_SIZE))))
model_entities.add(MaxPooling2D(dim_ordering='th'))
model_entities.add(Flatten())
model_entities.add(Dropout(0.5))

model_target = Sequential()
model_target.add(Conv2D(50, 3, 3, activation='relu', subsample=(3, 3), dim_ordering='th', input_shape=(1, (180 / GRID_SIZE), (360 / GRID_SIZE))))
model_target.add(MaxPooling2D(dim_ordering='th'))
model_target.add(Flatten())
model_target.add(Dropout(0.5))

merged_model = Sequential()
merged_model.add(Merge([left_pair, left_single, model_right_pair,
                        model_right_single, model_entities, model_target], mode='concat', concat_axis=1))
merged_model.add(Dense(output_dim=((180 / GRID_SIZE), (360 / GRID_SIZE)), activation='softmax'))
merged_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights_all_cnn", verbose=0)
# checkpoint = ModelCheckpoint(filepath="../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor='acc', patience=5)
file_name = u"data/eval_wiki.txt"
merged_model.fit_generator(generate_arrays_from_file(file_name, word_to_index, input_length, oneDim=False),
                           samples_per_epoch=int(check_output(["wc", file_name]).split()[0]),
                           nb_epoch=100, callbacks=[checkpoint, early_stop])

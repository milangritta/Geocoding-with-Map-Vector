# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Dense, Dropout, LSTM
from preprocessing import BATCH_SIZE, EMB_DIM, CONTEXT_LENGTH, UNKNOWN, TARGET_LENGTH, generate_arrays_from_file_lstm, \
    FILTER_2x2
from subprocess import check_output

print(u"Dimension:", EMB_DIM)
print(u"Input length:", CONTEXT_LENGTH)
#  --------------------------------------------------------------------------------------------------------------------
word_to_index = cPickle.load(open(u"data/w2i.pkl"))
print(u"Vocabulary Size:", len(word_to_index))

vectors = {UNKNOWN: np.ones(EMB_DIM), u'0': np.ones(EMB_DIM)}
for line in codecs.open(u"../data/glove.twitter." + str(EMB_DIM) + u"d.txt", encoding=u"utf-8"):
    if line.strip() == "":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]
print(u'Twitter vectors...', len(vectors))

weights = np.zeros((len(word_to_index), EMB_DIM))
oov = 0
for w in word_to_index:
    if w in vectors:
        weights[word_to_index[w]] = vectors[w]
    else:
        weights[word_to_index[w]] = np.random.normal(size=(EMB_DIM,), scale=0.3)
        oov += 1

weights = np.array([weights])
print(u'Done preparing vectors...')
print(u"OOV (no vectors):", oov)
#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
embeddings = Embedding(len(word_to_index), EMB_DIM, input_length=CONTEXT_LENGTH, weights=weights)
# shared embeddings between all language input layers

forward = Input(shape=(CONTEXT_LENGTH,))
cwf = embeddings(forward)
cwf = LSTM(300)(cwf)
cwf = Dense(300)(cwf)
cwf = Dropout(0.5)(cwf)

backward = Input(shape=(CONTEXT_LENGTH,))
cwb = embeddings(backward)
cwb = LSTM(300, go_backwards=True)(cwb)
cwb = Dense(300)(cwb)
cwb = Dropout(0.5)(cwb)

target_string = Input(shape=(TARGET_LENGTH,))
ts = Embedding(len(word_to_index), EMB_DIM, input_length=TARGET_LENGTH, weights=weights)(target_string)
ts = LSTM(50)(ts)
ts = Dense(50)(ts)
ts = Dropout(0.5)(ts)

inp = concatenate([cwf, cwb, ts])
inp = Dense(units=len(FILTER_2x2), activation=u'softmax')(inp)
model = Model(inputs=[forward, backward, target_string], outputs=[inp])
model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
# checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
checkpoint = ModelCheckpoint(filepath=u"../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor=u'acc', patience=5)
file_name = u"../data/train_wiki_uniform.txt"
model.fit_generator(generate_arrays_from_file_lstm(file_name, word_to_index),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=250, callbacks=[checkpoint, early_stop])

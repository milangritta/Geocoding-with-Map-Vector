# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Dense, Dropout, LSTM
from preprocessing import BATCH_SIZE, EMBEDDING_DIMENSION, CONTEXT_LENGTH, UNKNOWN, TARGET_LENGTH
from preprocessing import generate_arrays_from_file_lstm, ENCODING_MAP_2x2, ENCODING_MAP_1x1
from subprocess import check_output

print(u"Embedding Dimension:", EMBEDDING_DIMENSION)
print(u"Input length (each side):", CONTEXT_LENGTH)
word_to_index = cPickle.load(open(u"data/words2index.pkl"))
print(u"Vocabulary Size:", len(word_to_index))

vectors = {UNKNOWN: np.ones(EMBEDDING_DIMENSION), u'0': np.ones(EMBEDDING_DIMENSION)}
for line in codecs.open(u"../data/glove.twitter." + str(EMBEDDING_DIMENSION) + u"d.txt", encoding=u"utf-8"):
    if line.strip() == "":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]
print(u'Vectors...', len(vectors))

emb_weights = np.zeros((len(word_to_index), EMBEDDING_DIMENSION))
oov = 0
for w in word_to_index:
    if w in vectors:
        emb_weights[word_to_index[w]] = vectors[w]
    else:
        emb_weights[word_to_index[w]] = np.random.normal(size=(EMBEDDING_DIMENSION,), scale=0.3)
        oov += 1

emb_weights = np.array([emb_weights])
print(u'Done preparing vectors...')
print(u"OOV (no vectors):", oov)
#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
embeddings = Embedding(len(word_to_index), EMBEDDING_DIMENSION, input_length=CONTEXT_LENGTH, weights=emb_weights)
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

# Uncomment this block for MAPVEC + CONTEXT2VEC model, also uncomment 2 lines further down, thanks!
# You also need to uncomment a few lines in preprocessing.py, generate_arrays_from_file_lstm() function
# mapvec = Input(shape=(len(ENCODING_MAP_1x1),))
# l2v = Dense(5000, activation='relu', input_dim=len(ENCODING_MAP_1x1))(mapvec)
# l2v = Dense(1000, activation='relu')(l2v)
# l2v = Dropout(0.5)(l2v)

target_string = Input(shape=(TARGET_LENGTH,))
ts = Embedding(len(word_to_index), EMBEDDING_DIMENSION, input_length=TARGET_LENGTH, weights=emb_weights)(target_string)
ts = LSTM(50)(ts)
ts = Dense(50)(ts)
ts = Dropout(0.5)(ts)

inp = concatenate([cwf, cwb, ts])
# inp = concatenate([cwf, cwb, mapvec, ts])  # Uncomment for MAPVEC + CONTEXT2VEC
inp = Dense(units=len(ENCODING_MAP_2x2), activation=u'softmax')(inp)
model = Model(inputs=[forward, backward, target_string], outputs=[inp])
# model = Model(inputs=[forward, backward, mapvec, target_string], outputs=[inp])  # Uncomment for MAPVEC + CONTEXT2VEC
model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath=u"../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor=u'acc', patience=5)
file_name = u"../data/train_wiki_uniform.txt"
model.fit_generator(generate_arrays_from_file_lstm(file_name, word_to_index),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=250, callbacks=[checkpoint, early_stop])

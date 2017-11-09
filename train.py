# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from preprocessing import BATCH_SIZE, EMB_DIM, CONTEXT_LENGTH, UNKNOWN, PADDING
from preprocessing import TARGET_LENGTH, generate_arrays_from_file, FILTER_2x2, FILTER_1x1
from subprocess import check_output

print(u"Dimension:", EMB_DIM)
print(u"Input length:", CONTEXT_LENGTH)
#  --------------------------------------------------------------------------------------------------------------------
word_to_index = cPickle.load(open(u"data/w2i.pkl"))
print(u"Vocabulary Size:", len(word_to_index))

vectors = {UNKNOWN: np.ones(EMB_DIM), PADDING: np.ones(EMB_DIM)}
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
embeddings = Embedding(len(word_to_index), EMB_DIM, input_length=CONTEXT_LENGTH * 2, weights=weights)
# shared embeddings between all language input layers

context_words_pair = Input(shape=(CONTEXT_LENGTH * 2,))
cwp = embeddings(context_words_pair)
cwp = Conv1D(1000, 2, activation='relu', strides=1)(cwp)
cwp = GlobalMaxPooling1D()(cwp)
cwp = Dense(250)(cwp)
cwp = Dropout(0.5)(cwp)

context_words_single = Input(shape=(CONTEXT_LENGTH * 2,))
cws = embeddings(context_words_single)
cws = Conv1D(1000, 1, activation='relu', strides=1)(cws)
cws = GlobalMaxPooling1D()(cws)
cws = Dense(250)(cws)
cws = Dropout(0.5)(cws)

entities_strings_pair = Input(shape=(CONTEXT_LENGTH * 2,))
esp = embeddings(entities_strings_pair)
esp = Conv1D(1000, 2, activation='relu', strides=1)(esp)
esp = GlobalMaxPooling1D()(esp)
esp = Dense(250)(esp)
esp = Dropout(0.5)(esp)

entities_strings_single = Input(shape=(CONTEXT_LENGTH * 2,))
ess = embeddings(entities_strings_single)
ess = Conv1D(1000, 1, activation='relu', strides=1)(ess)
ess = GlobalMaxPooling1D()(ess)
ess = Dense(250)(ess)
ess = Dropout(0.5)(ess)

loc2vec = Input(shape=(len(FILTER_1x1),))
l2v = Dense(5000, activation='relu', input_dim=len(FILTER_1x1))(loc2vec)
l2v = Dense(1000, activation='relu')(l2v)
l2v = Dropout(0.5)(l2v)

target_string = Input(shape=(TARGET_LENGTH,))
ts = Embedding(len(word_to_index), EMB_DIM, input_length=TARGET_LENGTH, weights=weights)(target_string)
ts = Conv1D(1000, 3, activation='relu')(ts)
ts = GlobalMaxPooling1D()(ts)
ts = Dropout(0.5)(ts)

inp = concatenate([cwp, cws, esp, ess, l2v, ts])
inp = Dense(units=len(FILTER_2x2), activation=u'softmax')(inp)
model = Model(inputs=[context_words_pair, context_words_single, entities_strings_pair, entities_strings_single,
                      loc2vec, target_string], outputs=[inp])
model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
# checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
checkpoint = ModelCheckpoint(filepath=u"../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor=u'acc', patience=5)
file_name = u"../data/train_wiki_uniform.txt"
model.fit_generator(generate_arrays_from_file(file_name, word_to_index),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=250, callbacks=[checkpoint, early_stop])

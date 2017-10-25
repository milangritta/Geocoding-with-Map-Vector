# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from preprocessing import BATCH_SIZE, EMB_DIM, CONTEXT_LENGTH, UNKNOWN, PADDING, \
    TARGET_LENGTH, generate_arrays_from_file_loc
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
embeddings = Embedding(len(word_to_index), EMB_DIM, input_length=CONTEXT_LENGTH, weights=weights)
# shared embeddings between all language input layers

near_words = Input(shape=(CONTEXT_LENGTH,))
nw = embeddings(near_words)
nw = Conv1D(1000, 2, activation='relu', strides=1)(nw)
nw = GlobalMaxPooling1D()(nw)
nw = Dense(250)(nw)
nw = Dropout(0.5)(nw)

far_words = Input(shape=(CONTEXT_LENGTH,))
fw = embeddings(far_words)
fw = Conv1D(1000, 2, activation='relu', strides=1)(fw)
fw = GlobalMaxPooling1D()(fw)
fw = Dense(250)(fw)
fw = Dropout(0.5)(fw)

near_entities_strings = Input(shape=(CONTEXT_LENGTH,))
nes = embeddings(near_entities_strings)
nes = Conv1D(1000, 2, activation='relu', strides=1)(nes)
nes = GlobalMaxPooling1D()(nes)
nes = Dense(250)(nes)
nes = Dropout(0.5)(nes)

far_entities_strings = Input(shape=(CONTEXT_LENGTH,))
fes = embeddings(far_entities_strings)
fes = Conv1D(1000, 2, activation='relu', strides=1)(fes)
fes = GlobalMaxPooling1D()(fes)
fes = Dense(250)(fes)
fes = Dropout(0.5)(fes)

input_polygon_size = 2
near_entities_coord = Input(shape=((180 / input_polygon_size) * (360 / input_polygon_size),))
nec = Dense(250, activation='relu', input_dim=(180 / input_polygon_size) * (360 / input_polygon_size))(near_entities_coord)
nec = Dropout(0.5)(nec)

far_entities_coord = Input(shape=((180 / input_polygon_size) * (360 / input_polygon_size),))
fec = Dense(250, activation='relu', input_dim=(180 / input_polygon_size) * (360 / input_polygon_size))(far_entities_coord)
fec = Dropout(0.5)(fec)

target_coord = Input(shape=((180 / input_polygon_size) * (360 / input_polygon_size),))
tc = Dense(1000, activation='relu', input_dim=(180 / input_polygon_size) * (360 / input_polygon_size))(target_coord)
tc = Dropout(0.5)(tc)

target_string = Input(shape=(TARGET_LENGTH,))
ts = Embedding(len(word_to_index), EMB_DIM, input_length=TARGET_LENGTH, weights=weights)(target_string)
ts = Conv1D(1000, 3, activation='relu')(ts)
ts = GlobalMaxPooling1D()(ts)
ts = Dropout(0.5)(ts)

output_polygon_size = 2
inp = concatenate([nw, fw, nes, fes, nec, fec, tc, ts])
inp = Dense(units=(180 / output_polygon_size) * (360 / output_polygon_size), activation=u'softmax')(inp)
model = Model(inputs=[near_words, far_words, near_entities_strings, far_entities_strings,
                      near_entities_coord, far_entities_coord, target_coord, target_string], outputs=[inp])
model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
# checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
checkpoint = ModelCheckpoint(filepath=u"../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor=u'acc', patience=5)
file_name = u"../data/train_wiki_uniform.txt"
model.fit_generator(generate_arrays_from_file_loc(file_name),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=200, callbacks=[checkpoint, early_stop])

# -*- coding: utf-8 -*-
import cPickle
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Input, concatenate
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from preprocessing import generate_arrays_from_file, GRID_SIZE, BATCH_SIZE, CONTEXT_LENGTH, EMB_DIM
from subprocess import check_output

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# set_session(tf.Session(config=config))

print(u"Dimension:", EMB_DIM)
print(u"Input length:", CONTEXT_LENGTH)

vocabulary = cPickle.load(open(u"data/vocabulary.pkl"))
print(u"Vocabulary Size:", len(vocabulary))
#  --------------------------------------------------------------------------------------------------------------------
print(u'Preparing vectors...')
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])

# vectors = {UNKNOWN: np.ones(dimension), PADDING: np.ones(dimension)}
# for line in codecs.open(u"../data/glove.twitter." + str(dimension) + "d.txt", encoding="utf-8"):
#     if line.strip() == u"":
#         continue
#     t = line.split()
#     vectors[t[0]] = [float(x) for x in t[1:]]
# print(u'Loaded Twitter vectors...', len(vectors))

# for line in codecs.open(u"../data/glove." + str(dimension) + "d.txt", encoding="utf-8"):
#     if line.strip() == u"":
#         continue
#     t = line.split()
#     vectors[t[0]] = [float(x) for x in t[1:]]
# print(u'Loaded GloVe vectors...', len(vectors))

# weights = np.zeros((len(vocabulary), dimension))
# for w in vocabulary:
#     if w in vectors:
#         weights[word_to_index[w]] = vectors[w]
# weights = np.array([weights])
# print(u'Done preparing vectors...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')

entities = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
ent = Conv2D(50, (5, 5), strides=(4, 4), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(entities)
ent = Conv2D(100, (4, 4), strides=(3, 3), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(ent)
ent = MaxPooling2D(data_format="channels_first")(ent)
ent = Flatten()(ent)
ent = Dense(250)(ent)

target = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
tar = Conv2D(50, (2, 2), strides=(2, 2), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(target)
tar = Conv2D(100, (2, 2), strides=(2, 2), activation="relu", data_format="channels_first",
             input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))(tar)
tar = MaxPooling2D(data_format="channels_first")(tar)
tar = Flatten()(tar)
tar = Dense(250)(tar)

merged = concatenate([ent, tar])
merged = Dense(units=(180 / GRID_SIZE) * (360 / GRID_SIZE), activation='softmax')(merged)
model = Model(inputs=[entities, target], outputs=[merged])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights_all_cnn", verbose=0)
# checkpoint = ModelCheckpoint(filepath="../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
# early_stop = EarlyStopping(monitor='acc', patience=5)
file_name = u"data/eval_wiki.txt"
model.fit_generator(generate_arrays_from_file(file_name, word_to_index, oneDim=False),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=100, callbacks=[checkpoint])

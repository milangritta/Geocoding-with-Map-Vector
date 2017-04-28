import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Activation, Concatenate
from keras.models import Sequential
from preprocessing import pad_list, construct_1D_grid, construct_2D_grid

print(u'Loading training data...')
X_L, X_R, X_E, X_T, Y, N = [], [], [], [], [], []
UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 50, 50
vocabulary = cPickle.load(open("data/vocabulary.pkl"))

training_file = codecs.open("./data/eval_lgl.txt", "r", encoding="utf-8")
for line in training_file:
    line = line.strip().split("\t")
    Y.append(construct_1D_grid([(float(line[0]), float(line[1]), 0)], False))
    X_L.append(pad_list(input_length, eval(line[2].lower()), True))
    X_R.append(pad_list(input_length, eval(line[3].lower()), False))
    X_E.append(construct_2D_grid(eval(line[4]), False))
    X_T.append(construct_2D_grid(eval(line[5]), True))
    N.append(line[6])

print(u"Vocabulary Size:", len(vocabulary))
print(u"No of training examples:", len(N))
#  --------------------------------------------------------------------------------------------------------------------
print(u'Preparing vectors...')
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])

for x_l, x_r in zip(X_L, X_R):
    for i, w in enumerate(x_l):
        if w in word_to_index:
            x_l[i] = word_to_index[w]
        else:
            x_l[i] = word_to_index[UNKNOWN]
    for i, w in enumerate(x_r):
        if w in word_to_index:
            x_r[i] = word_to_index[w]
        else:
            x_r[i] = word_to_index[UNKNOWN]

X_L = np.asarray(X_L)
X_R = np.asarray(X_R)
X_E = np.asarray(X_E)
X_T = np.asarray(X_T)
Y = np.asarray(Y)

vectors = {UNKNOWN: np.ones(50)}
for line in codecs.open("../data/glove.twitter.50d.txt", encoding="utf-8"):
    if line.strip() == "":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]

weights = np.zeros((len(vocabulary), 50))
for w in vocabulary:
    if w in vectors:
        weights[word_to_index[w]] = vectors[w]
weights = np.array([weights])

#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
model_left = Sequential()
model_left.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_left.add(Conv1D(250, 2, padding='valid', activation='relu', strides=1))
model_left.add(GlobalMaxPooling1D())
model_left.add(Dense(25))
model_left.add(Dropout(0.2))
model_left.add(Activation('relu'))

model_right = Sequential()
model_right.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_right.add(Conv1D(250, 2, padding='valid', activation='relu', strides=1))
model_right.add(GlobalMaxPooling1D())
model_right.add(Dense(25))
model_right.add(Dropout(0.2))
model_right.add(Activation('relu'))

model_target = Sequential()
model_target.add(Conv1D(250, 2, padding='valid', activation='relu', strides=1, input_shape=(36, 72)))
model_target.add(GlobalMaxPooling1D())
model_target.add(Dense(250))
model_target.add(Dropout(0.2))
model_target.add(Activation('relu'))

model_entities = Sequential()
model_entities.add(Conv1D(250, 2, padding='valid', activation='relu', strides=1, input_shape=(36, 72)))
model_entities.add(GlobalMaxPooling1D())
model_entities.add(Dense(25))
model_entities.add(Dropout(0.2))
model_entities.add(Activation('relu'))

merged_model = Sequential()
concat = Concatenate(axis=1, input_shape=(4, 25))
concat([model_left.layers[-1].output, model_right.layers[-1].output, model_target.layers[-1].output, model_entities.layers[-1].output])
merged_model.add(concat)
merged_model.add(Dense(25))
merged_model.add(Dense(output_dim=36*72, activation='softmax'))
merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------

checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
merged_model.fit([X_L, X_R, X_T, X_E], Y, batch_size=64, epochs=50, callbacks=[checkpoint], verbose=1)
# first_input = Input(shape=(2, ))
# first_dense = Dense(1, )(first_input)
#
# second_input = Input(shape=(2, ))
# second_dense = Dense(1, )(second_input)
#
# merge_one = concatenate([first_dense, second_dense])
#
# third_input = Input(shape=(1, ))
# merge_two = concatenate([merge_one, third_input])
#
# model = Model(inputs=[first_input, second_input, third_input], outputs=merge_two)
# result.compile(optimizer=ada_grad, loss='binary_crossentropy', metrics=['accuracy']

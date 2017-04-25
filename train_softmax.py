import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint
from keras.engine import Merge
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from preprocessing import pad_list, construct_1D_grid

print(u'Loading training data...')
X_L, X_R, X_E, X_T, Y, N = [], [], [], [], [], []
UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 50, 50
vocabulary = cPickle.load(open("./data/vocabulary.pkl"))

training_file = codecs.open("./data/output.txt", "r", encoding="utf-8")
for line in training_file:
    line = line.strip().split("\t")
    Y.append(construct_1D_grid([(float(line[0]), float(line[1]), 0)], False))
    X_L.append(pad_list(input_length, eval(line[2].lower()), True))
    X_R.append(pad_list(input_length, eval(line[3].lower()), False))
    X_E.append(construct_1D_grid(eval(line[4]), True))
    X_T.append(construct_1D_grid(eval(line[5]), False))
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
for line in codecs.open("data/glove.twitter.50d.txt", encoding="utf-8"):
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
model_left.add(LSTM(output_dim=50))
model_left.add(Dropout(0.2))

model_right = Sequential()
model_right.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_right.add(LSTM(output_dim=50, go_backwards=True))
model_right.add(Dropout(0.2))

model_target = Sequential()
model_target.add(Dense(output_dim=100, activation='relu', input_dim=36*72))
model_target.add(Dropout(0.2))
model_target.add(Dense(output_dim=50, activation='relu'))

model_entities = Sequential()
model_entities.add(Dense(output_dim=100, activation='relu', input_dim=36*72))
model_entities.add(Dropout(0.2))
model_entities.add(Dense(output_dim=50, activation='relu'))

merged_model = Sequential()
merged_model.add(Merge([model_left, model_right, model_target, model_entities], mode='concat', concat_axis=1))
merged_model.add(Dense(output_dim=25))
merged_model.add(Dense(output_dim=36*72, activation='softmax'))
merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------

checkpoint = ModelCheckpoint(filepath="./data/lstm.weights", verbose=0)
merged_model.fit([X_L, X_R, X_T, X_E], Y, batch_size=64, nb_epoch=50, callbacks=[checkpoint], verbose=1)

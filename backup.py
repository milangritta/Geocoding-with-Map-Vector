import codecs
import numpy as np
import cPickle
from keras.callbacks import ModelCheckpoint
from keras.engine import Merge
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from preprocessing import pad_list, construct_1D_grid, GRID_SIZE

print(u'Loading training data...')
X_L, X_R, X_E, X_T, Y, N = [], [], [], [], [], []
UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 50, 50
vocabulary = cPickle.load(open("data/vocabulary.pkl"))

training_file = codecs.open("data/eval_lgl.txt", "r", encoding="utf-8")
for line in training_file:
    line = line.strip().split("\t")
    Y.append(construct_1D_grid([(float(line[0]), float(line[1]), 0)], use_pop=False))
    X_L.append(pad_list(input_length, eval(line[2].lower()), from_left=True))
    X_R.append(pad_list(input_length, eval(line[3].lower()), from_left=False))
    X_E.append(construct_1D_grid(eval(line[4]), use_pop=False))
    X_T.append(construct_1D_grid(eval(line[5]), use_pop=True))
    N.append(line[6])

print(u"Vocabulary Size:", len(vocabulary))
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

vectors = {UNKNOWN: np.ones(dimension)}
for line in codecs.open("../data/glove.twitter." + str(dimension) + "d.txt", encoding="utf-8"):
    if line.strip() == "":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]

weights = np.zeros((len(vocabulary), dimension))
for w in vocabulary:
    if w in vectors:
        weights[word_to_index[w]] = vectors[w]
weights = np.array([weights])

#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
model_left = Sequential()
model_left.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_left.add(LSTM(25))
model_left.add(Dropout(0.2))

model_right = Sequential()
model_right.add(Embedding(len(vocabulary), dimension, input_length=input_length, weights=weights))
model_right.add(LSTM(25, go_backwards=True))
model_right.add(Dropout(0.2))

model_target = Sequential()
model_target.add(Dense(500, activation='relu', input_dim=(180 / GRID_SIZE) * (360 / GRID_SIZE)))
model_target.add(Dropout(0.2))
model_target.add(Dense(250, activation='relu'))

model_entities = Sequential()
model_entities.add(Dense(50, activation='relu', input_dim=(180 / GRID_SIZE) * (360 / GRID_SIZE)))
model_entities.add(Dropout(0.2))
model_entities.add(Dense(25, activation='relu'))

merged_model = Sequential()
merged_model.add(Merge([model_left, model_right, model_target, model_entities], mode='concat', concat_axis=1))
merged_model.add(Dense(25))
merged_model.add(Dense((180 / GRID_SIZE) * (360 / GRID_SIZE), activation='softmax'))
merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
merged_model.fit([X_L, X_R, X_T, X_E], Y, batch_size=128, nb_epoch=100, callbacks=[checkpoint], verbose=1)

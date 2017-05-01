import codecs
import numpy as np
import cPickle
import sqlite3
from geopy.distance import great_circle
from keras.models import load_model
from preprocessing import pad_list, construct_1D_grid, get_coordinates, print_stats, index_to_coord, GRID_SIZE
import matplotlib.pyplot as plt

print(u'Loading training data...')
X_L, X_R, X_E, X_T, N, C = [], [], [], [], [], []
UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 50, 50
vocabulary = cPickle.load(open("./data/vocabulary.pkl"))

training_file = codecs.open("./data/eval_lgl.txt", "r", encoding="utf-8")
for line in training_file:
    line = line.strip().split("\t")
    C.append((float(line[0]), float(line[1])))
    X_L.append(pad_list(input_length, eval(line[2].lower()), from_left=True))
    X_R.append(pad_list(input_length, eval(line[3].lower()), from_left=False))
    X_E.append(construct_1D_grid(eval(line[4]), use_pop=False))
    X_T.append(construct_1D_grid(eval(line[5]), use_pop=True))
    N.append(line[6])

print(u"Vocabulary Size:", len(vocabulary))
print(u'Loaded', len(C), u'test examples.')
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

#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(u"../data/weights")
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
conn = sqlite3.connect(u'../data/geonames.db')
choice = []
for p, c, n, e in zip(model.predict([X_L, X_R, X_T, X_E]), C, N, X_E):
    p = index_to_coord(np.argmax(p))
    candidates = eval(get_coordinates(conn.cursor(), n))
    if len(candidates) == 0:
        print(u"Don't have an entry for", n, u"in GeoNames")
        continue
    temp = []
    for candidate in candidates:
        temp.append((great_circle(p, (float(candidate[0]), float(candidate[1]))).kilometers, (float(candidate[0]), float(candidate[1]))))
    best = sorted(temp, key=lambda (x, y): x)[0]
    choice.append(great_circle(best[1], c).kilometers)
    print(n, p, c, choice[-1])
    print(candidates, sorted(temp)[0])
    print("-----------------------------------------------------------------------------------------------------------")


print_stats(choice)
plt.plot(range(len(choice)), sorted(choice))
plt.xlabel(u"Examples")
plt.ylabel(u'Error')
plt.show()

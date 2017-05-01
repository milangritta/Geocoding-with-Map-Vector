import numpy as np
import cPickle
import sqlite3
from geopy.distance import great_circle
from keras.models import load_model
from subprocess import check_output
from preprocessing import get_coordinates, print_stats, index_to_coord, generate_names_from_file
from preprocessing import generate_arrays_from_file
from preprocessing import generate_labels_from_file
# import matplotlib.pyplot as plt

print(u'Loading test data...')
UNKNOWN, PADDING = u"<unknown>", u"0.0"
dimension, input_length = 50, 50

vocabulary = cPickle.load(open(u"./data/vocabulary.pkl"))
print(u"Vocabulary Size:", len(vocabulary))
#  --------------------------------------------------------------------------------------------------------------------
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])
#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(u"../data/weights")
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
conn = sqlite3.connect(u'../data/geonames.db')
file_name = u"data/eval_lgl.txt"
choice = []
for p, y, n in zip(model.predict_generator(generate_arrays_from_file(file_name, word_to_index, train=False),
                   val_samples=int(check_output(["wc", file_name]).split()[0])),
                   generate_labels_from_file(file_name), generate_names_from_file(file_name)):
    p = index_to_coord(np.argmax(p))
    candidates = eval(get_coordinates(conn.cursor(), n))
    if len(candidates) == 0:
        print(u"Don't have an entry for", n, u"in GeoNames")
        continue
    temp = []
    for candidate in candidates:
        temp.append((great_circle(p, (float(candidate[0]), float(candidate[1]))).kilometers, (float(candidate[0]), float(candidate[1]))))
    best = sorted(temp, key=lambda (a, b): a)[0]
    choice.append(great_circle(best[1], y).kilometers)
    print(n, p, y, choice[-1])
    print(candidates, sorted(temp)[0])
    print("-----------------------------------------------------------------------------------------------------------")

print_stats(choice)
# plt.plot(range(len(choice)), sorted(choice))
# plt.xlabel(u"Examples")
# plt.ylabel(u'Error')
# plt.show()

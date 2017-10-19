# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
import sqlite3
import sys
from geopy.distance import great_circle
from keras.models import load_model
from subprocess import check_output
from preprocessing import get_coordinates, print_stats, index_to_coord, generate_strings_from_file, CONTEXT_LENGTH, BATCH_SIZE
from preprocessing import generate_arrays_from_file
# import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    data = u"lgl"

weights_file = u"../data/weights"
print(u"Input length:", CONTEXT_LENGTH)
print(u"Testing:", data, u"with weights:", weights_file)
word_to_index = cPickle.load(open(u"data/w2i.pkl"))
#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(weights_file)
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Crunching numbers, sit tight...')
errors = codecs.open(u"errors.tsv", u"w", encoding=u"utf-8")
conn = sqlite3.connect(u'../data/geonames.db')
file_name = u"data/eval_" + data + u".txt"
final_errors = []
for p, (y, name, context) in zip(model.predict_generator(generate_arrays_from_file(file_name, word_to_index, train=False),
                                 steps=int(check_output([u"wc", file_name]).split()[0]) / BATCH_SIZE, verbose=True),
                                 generate_strings_from_file(file_name)):
    p = index_to_coord(np.argmax(p))
    candidates = get_coordinates(conn.cursor(), name)
    candidates = sorted(candidates, key=lambda (a, b, c, d): c, reverse=True)

    if len(candidates) == 0:
        print(u"Don't have an entry for", name, u"in GeoNames")
        raise Exception(u"Go back and check your geo-database, buddy :-)")

    # candidates = [candidates[0]]  # Uncomment for population heuristic.
    # THE ABOVE IS THE POPULATION ONLY BASELINE IMPLEMENTATION

    best_candidate, y_to_geonames = [], []
    # max_pop = candidates[0][2]
    for candidate in candidates:
        y_to_geonames.append(great_circle(y, (float(candidate[0]), float(candidate[1]))).km)
        best_candidate.append((great_circle(p, (float(candidate[0]), float(candidate[1]))).km, (float(candidate[0]), float(candidate[1]))))
    best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]
    final_errors.append(great_circle(best_candidate[1], y).km)

    # ---------------- DIAGNOSTICS --------------------
    dist_p, dist_y, index_p, index_y = 50000, 50000, 0, 0
    for index, candidate in enumerate(candidates):
        if great_circle(best_candidate[1], (candidate[0], candidate[1])).km < dist_p:
            dist_p = great_circle(best_candidate[1], (candidate[0], candidate[1])).km
            index_p = index
        if great_circle(y, (candidate[0], candidate[1])).km < dist_y:
            dist_y = great_circle(y, (candidate[0], candidate[1])).km
            index_y = index
    # ------------- END OF DIAGNOSTICS -----------------

    errors.write(name + u"\t" + unicode(y) + u"\t" + unicode(p) + "\t" + unicode(best_candidate[1])
                 + u"\t" + unicode(index_y) + u"\t" + unicode(index_p) + u"\t" + unicode(final_errors[-1]) + u"\t" +
                 unicode(best_candidate[0]) + u"\t" + unicode(len(candidates)) + u"\t" + unicode(sorted(y_to_geonames)[0])
                 + u"\t" + context.replace(u"\n", u"") + u"\n")
    # print("-----------------------------------------------------------------------------------------------------------")

print_stats(final_errors)
print(u"Processed file", file_name)

# ---------------- DIAGNOSTICS --------------------
# plt.plot(range(len(choice)), np.log(1 + np.asarray(sorted(choice))))
# plt.xlabel(u"Predictions")
# plt.ylabel(u'Error Size')
# plt.title(u"Some Chart")
# plt.savefig(u'test.png', transparent=True)
# plt.show()

# W = model.layers[-1].get_weights()
# W = np.concatenate((W[0], np.array([W[1]])), axis=0)
# W = np.rot90(W)
# cPickle.dump(W, open("./data/W.pkl", "w"))
# ------------- END OF DIAGNOSTICS -----------------

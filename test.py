# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
import sqlite3
import sys
from geopy.distance import great_circle
from keras.models import load_model
from subprocess import check_output
from preprocessing import get_coordinates, print_stats, index_to_coord, generate_strings_from_file
from preprocessing import BATCH_SIZE, REVERSE_MAP_2x2
from preprocessing import generate_arrays_from_file

# For command line use, type: python test.py <dataset name>
# For example: python test.py lgl_gold
if len(sys.argv) > 1:
    test_data = sys.argv[1]
else:
    test_data = u"geovirus"  # or edit this line if running inside an IDE editor

saved_model_file = u"../data/weights"
print(u"Testing:", test_data, u"with weights:", saved_model_file)
word_to_index = cPickle.load(open(u"data/words2index.pkl"))  # This is the vocabulary file
#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(saved_model_file)
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Crunching numbers, sit tight...')
# errors = codecs.open(u"errors.tsv", u"w", encoding=u"utf-8")
# Uncomment the above line for error diagnostics, also the section below.
conn = sqlite3.connect(u'../data/geonames.db')
file_name = u"data/eval_" + test_data + u".txt"
final_errors = []
for prediction, (y, name, context) in zip(model.predict_generator(generate_arrays_from_file(file_name, word_to_index, train=False),
                                          steps=int(check_output([u"wc", file_name]).split()[0]) / BATCH_SIZE, verbose=True), generate_strings_from_file(file_name)):
    prediction = index_to_coord(REVERSE_MAP_2x2[np.argmax(prediction)], 2)
    candidates = get_coordinates(conn.cursor(), name)

    if len(candidates) == 0:
        print(u"Don't have an entry for", name, u"in GeoNames")
        raise Exception(u"Check your database, buddy :-)")

    # candidates = [candidates[0]]  # Uncomment for population heuristic.
    # THE ABOVE IS THE POPULATION ONLY BASELINE IMPLEMENTATION

    best_candidate = []
    max_pop = candidates[0][2]
    bias = 0.9  # the Bias parameter in the paper
    for candidate in candidates:
        err = great_circle(prediction, (float(candidate[0]), float(candidate[1]))).km
        best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * bias, (float(candidate[0]), float(candidate[1]))))
    best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]
    final_errors.append(great_circle(best_candidate[1], y).km)

    # ---------------- ERROR DIAGNOSTICS --------------------
    # dist_p, dist_y, index_p, index_y = 100000, 100000, 0, 0
    # for index, candidate in enumerate(candidates):
    #     if great_circle(best_candidate[1], (candidate[0], candidate[1])).km < dist_p:
    #         dist_p = great_circle(best_candidate[1], (candidate[0], candidate[1])).km
    #         index_p = index
    #     if great_circle(y, (candidate[0], candidate[1])).km < dist_y:
    #         dist_y = great_circle(y, (candidate[0], candidate[1])).km
    #         index_y = index
    #
    # errors.write(name + u"\t" + unicode(y) + u"\t" + unicode(p) + "\t" + unicode(best_candidate[1])
    #              + u"\t" + unicode(index_y) + u"\t" + unicode(index_p) + u"\t" + unicode(final_errors[-1]) + u"\t" +
    #              unicode(best_candidate[0]) + u"\t" + unicode(len(candidates)) + u"\t" + context.replace(u"\n", u"") + u"\n")
    # ------------------ END OF DIAGNOSTICS -----------------

print_stats(final_errors)
print(u"Processed file", file_name)

# ------------------------ VISUALISATION ------------------------------
# import matplotlib.pyplot as plt
# plt.plot(range(len(final_errors)), np.log(1 + np.asarray(sorted(final_errors))))
# plt.xlabel(u"Predictions")
# plt.ylabel(u'Error Size')
# plt.title(u"Some Chart")
# plt.savefig(u'test.png', transparent=True)
# plt.show()
# ----------------------------------------------------------------------

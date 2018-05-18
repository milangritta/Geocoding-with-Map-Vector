import sqlite3
import sys
from geopy.distance import great_circle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from preprocessing import generate_arrays_from_file_map2vec, index_to_coord, get_coordinates, generate_strings_from_file
from preprocessing import REVERSE_MAP_2x2
from preprocessing import print_stats
import numpy as np
from sklearn.externals import joblib

# For command line use, type: python test.py <dataset name> such as lgl_gold or wiki (see file names)
if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    data = u"lgl"

X, Y = [], []
clf = MultinomialNB()
classes = range(len(REVERSE_MAP_2x2))
# clf = RandomForestClassifier()
for (x, y) in generate_arrays_from_file_map2vec(u"../data/train_wiki_uniform.txt", looping=False):
    X.extend(x[0])
    Y.extend(np.argmax(y, axis=1))
    # -------- Uncomment for Naive Bayes -------------
    if len(X) > 25000:
        print(u"Training with:", len(X), u"examples.")
        clf.partial_fit(X, Y, classes)
        X, Y = [], []
    # ------------------------------------------------

print(u"Training with:", len(X), u"examples.")
clf.partial_fit(X, Y, classes)  # Naive Bayes only!
# clf.fit(X, Y)  # Random Forest
joblib.dump(clf, u'../data/bayes.pkl')  # saves the model to file

# ------------------------------------- END OF TRAINING, BEGINNING OF TESTING -----------------------------------

X = []
final_errors = []
clf = joblib.load(u'../data/bayes.pkl')
test_file = u"data/eval_" + data + u".txt"  # which data to test on?

for (x, y) in generate_arrays_from_file_map2vec(test_file, looping=False):
    X.extend(x[0])  # Load test instances

print(u"Testing with:", len(X), u"examples.")
conn = sqlite3.connect(u'../data/geonames.db')

for x, (y, name, context) in zip(clf.predict(X), generate_strings_from_file(test_file)):
    p = index_to_coord(REVERSE_MAP_2x2[x], 2)
    candidates = get_coordinates(conn.cursor(), name)

    if len(candidates) == 0:
        print(u"Don't have an entry for", name, u"in GeoNames")
        raise Exception(u"Check your database, buddy :-)")

    # candidates = [candidates[0]]  # Uncomment for population heuristic.
    # THE ABOVE IS THE POPULATION ONLY BASELINE IMPLEMENTATION

    best_candidate = []
    max_pop = candidates[0][2]
    bias = 0.9  # bias parameter, see
    for candidate in candidates:
        err = great_circle(p, (float(candidate[0]), float(candidate[1]))).km
        best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * bias, (float(candidate[0]), float(candidate[1]))))
    best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]
    final_errors.append(great_circle(best_candidate[1], y).km)

print_stats(final_errors)
print(u"Done testing:", test_file)

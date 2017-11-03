import sqlite3
from geopy.distance import great_circle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from preprocessing import generate_arrays_from_file_loc, index_to_coord, get_coordinates, generate_strings_from_file
from preprocessing import REVERSE_2x2
from preprocessing import print_stats
import numpy as np
from sklearn.externals import joblib

X, Y = [], []
# clf = MultinomialNB()
clf = RandomForestClassifier()
for (x, y) in generate_arrays_from_file_loc(u"data/eval_lgl_gold.txt", looping=False):
    X.extend(x[0])
    Y.extend(np.argmax(y, axis=1))

print(u"Fitting with:", len(X), u"examples.")
clf.fit(X, Y)
joblib.dump(clf, u'../data/bayes.pkl')

# ------------------------------------- END OF TRAINING, BEGINNING OF TESTING -----------------------------------

X = []
final_errors = []
clf = joblib.load(u'../data/bayes.pkl')
test_file = u"data/eval_lgl.txt"

for (x, y) in generate_arrays_from_file_loc(test_file, looping=False):
    X.extend(x[0])

print(u"Testing with:", len(X), u"examples.")
conn = sqlite3.connect(u'../data/geonames.db')

for x, (y, name, context) in zip(clf.predict(X), generate_strings_from_file(test_file)):
    p = index_to_coord(REVERSE_2x2[x], 2)
    candidates = get_coordinates(conn.cursor(), name)
    candidates = sorted(candidates, key=lambda (a, b, c, d): c, reverse=True)

    if len(candidates) == 0:
        print(u"Don't have an entry for", name, u"in GeoNames")
        raise Exception(u"Go back and check your geo-database, buddy :-)")

    # candidates = [candidates[0]]  # Uncomment for population heuristic.
    # THE ABOVE IS THE POPULATION ONLY BASELINE IMPLEMENTATION

    best_candidate, y_to_geonames = [], []
    max_pop = candidates[0][2]
    for candidate in candidates:
        y_to_geonames.append(great_circle(y, (float(candidate[0]), float(candidate[1]))).km)
        err = great_circle(p, (float(candidate[0]), float(candidate[1]))).km
        # best_candidate.append((err, (float(candidate[0]), float(candidate[1]))))
        best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * 0.9, (float(candidate[0]), float(candidate[1]))))
    best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]
    final_errors.append(great_circle(best_candidate[1], y).km)

print_stats(final_errors)
print(u"Done testing:", test_file)

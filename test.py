# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import sqlite3
# import pprint
import sys
from geopy.distance import great_circle
from keras.models import load_model
from subprocess import check_output
from preprocessing import get_coordinates, print_stats, index_to_coord, generate_strings_from_file, GRID_SIZE
from preprocessing import generate_arrays_from_file, visualise_2D_grid, coord_to_index
# import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    data = sys.argv[1]
else:
    data = u"wiki"

input_length = 200
print(u"Input length:", input_length)
print(u"Testing:", data)
vocabulary = cPickle.load(open(u"./data/vocabulary.pkl"))
print(u"Vocabulary Size:", len(vocabulary))
#  --------------------------------------------------------------------------------------------------------------------
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])
#  --------------------------------------------------------------------------------------------------------------------
print(u'Loading model...')
model = load_model(u"../data/weights")
print(u'Finished loading model...')
#  --------------------------------------------------------------------------------------------------------------------
print(u'Crunching numbers, sit tight...')
conn = sqlite3.connect(u'../data/geonames.db')
file_name = u"data/eval_" + data + u".txt"
choice = []
for p, (y, name, context) in zip(model.predict_generator(generate_arrays_from_file(file_name, word_to_index, input_length, train=False),
                   val_samples=int(check_output(["wc", file_name]).split()[0])), generate_strings_from_file(file_name)):

    # ------------ DIAGNOSTICS ----------------
    # sort = p.argsort()[-10:]
    # print(coord_to_index(y), name)
    # for s in sort:
    #     print(s, int(s / (360 / GRID_SIZE)), s % (360 / GRID_SIZE), p[s])
    # new_p = np.copy(p)
    # new_p[coord_to_index(y)] += 1
    # new_p = np.reshape(new_p, (180 / GRID_SIZE, 360 / GRID_SIZE))
    # com = center_of_mass(np.reshape(p, (180 / GRID_SIZE, 360 / GRID_SIZE)))
    # new_p[int(com[0]), int(com[1])] += 1
    # visualise_2D_grid(new_p, name)
    # print()
    # --------- END OF DIAGNOSTICS -------------

    p = index_to_coord(np.argmax(p))
    candidates = get_coordinates(conn.cursor(), name, pop_only=True)

    # candidates = [sorted(get_coordinates(conn.cursor(), name, True), key=lambda (a, b, c): c, reverse=True)[0]]
    # THE ABOVE IS THE POPULATION ONLY BASELINE IMPLEMENTATION

    if len(candidates) == 0:
        print(u"Don't have an entry for", name, u"in GeoNames")
        continue
    temp, distance = [], []
    for candidate in candidates:
        distance.append((great_circle(y, (float(candidate[0]), float(candidate[1]))).kilometers, (float(candidate[0]), float(candidate[1]))))
        temp.append((great_circle(p, (float(candidate[0]), float(candidate[1]))).kilometers, (float(candidate[0]), float(candidate[1]))))
    best = sorted(temp, key=lambda (a, b): a)[0]
    choice.append(great_circle(best[1], y).kilometers)

    print(name, u"Predicted:", p, u"Gold:", y, u"Distance:", choice[-1])
    if sorted(distance)[0][0] > 161:
        print(u"OMW! No (GOLD < 161km) GeoNames entry for", name, u"Gold:", y, u"Predicted:", p)
        print(u"Best GeoNames Candidate:", sorted(distance, key=lambda (a, b): a)[0], u"My Distance:", choice[-1])
    print("-----------------------------------------------------------------------------------------------------------")

print_stats(choice)
print(u"Processed file", file_name)

# ---------------- DIAGNOSTICS --------------------
# pprint.pprint(model.get_config())
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

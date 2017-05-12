# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import numpy as np
import sqlite3
from matplotlib import pyplot, colors

GRID_SIZE = 2


def print_stats(accuracy):
    """"""
    print("==============================================================================================")
    accuracy = np.log(np.array(accuracy) + 1)
    print(u"Median error:", np.median(sorted(accuracy)))
    print(u"Mean error:", np.mean(accuracy))
    k = np.log(161)  # This is the k in accuracy@k metric (see my Survey Paper for details)
    print u"Accuracy to 161 km: ", sum([1.0 for dist in accuracy if dist < k]) / len(accuracy)
    print u"AUC = ", np.trapz(accuracy) / (np.log(20039) * (len(accuracy) - 1))  # Trapezoidal rule.
    print("==============================================================================================")


def pad_list(size, a_list, from_left):
    """"""
    while len(a_list) < size:
        if from_left:
            a_list = [0.0] + a_list
        else:
            a_list += [0.0]
    return a_list


def coord_to_index(coordinates, get_xy):
    """"""
    latitude = float(coordinates[0]) - 90 if float(coordinates[0]) != -90 else -179.99
    longitude = float(coordinates[1]) + 180 if float(coordinates[1]) != 180 else 359.99
    if longitude < 0:
        longitude = -longitude
    if latitude < 0:
        latitude = -latitude
    if not get_xy:
        return latitude, longitude
    x = (360 / GRID_SIZE) * (int(latitude) / GRID_SIZE)
    y = int(longitude) / GRID_SIZE
    return x + y if 0 <= x + y <= (360 / GRID_SIZE) * (180 / GRID_SIZE) else Exception(u"Shock horror!!")


def index_to_coord(index):
    """"""
    x = int(index / (360 / GRID_SIZE))
    y = index % (360 / GRID_SIZE)
    if x > (90 / GRID_SIZE):
        x = -(x - (90 / GRID_SIZE)) * GRID_SIZE
    else:
        x = ((90 / GRID_SIZE) - x) * GRID_SIZE
    if y < (180 / GRID_SIZE):
        y = -((180 / GRID_SIZE) - y) * GRID_SIZE
    else:
        y = (y - (180 / GRID_SIZE)) * GRID_SIZE
    return x, y


def get_coordinates(con, loc_name, pop_only):
    """"""
    result = con.execute(u"SELECT * FROM GEO WHERE NAME = ?", (loc_name.lower(), )).fetchone()
    if result:
        result = eval(result[1])
        if pop_only:
            if max(result, key=lambda(a, b, c): c)[2] == 0:
                return result
            else:
                return [r for r in result if r[2] > 0]
        else:
            return result
    else:
        return []


def construct_1D_grid(a_list, use_pop):
    """"""
    g = np.zeros((360 / GRID_SIZE) * (180 / GRID_SIZE))
    for s in a_list:
        if use_pop:
            g[coord_to_index((s[0], s[1]), True)] += 1 + s[2]
        else:
            g[coord_to_index((s[0], s[1]), True)] += 1
    return g / max(g) if max(g) > 0.0 else g


def construct_2D_grid(a_list, use_pop):
    """"""
    g = np.zeros(((180 / GRID_SIZE), (360 / GRID_SIZE)))
    for s in a_list:
        x, y = coord_to_index((s[0], s[1]), False)
        x = int(x / GRID_SIZE)
        y = int(y / GRID_SIZE)
        if use_pop:
            g[x][y] += 1 + s[2]
        else:
            g[x][y] += 1
    return g / np.amax(g) if np.amax(g) > 0.0 else g


def merge_lists(grids):
    """"""
    out = []
    for g in grids:
        out.extend(list(g))
    return out


def populate_geosql():
    """Create and populate the sqlite database with GeoNames data"""
    geo_names = {}
    f = codecs.open("../data/allCountries.txt", "r", encoding="utf-8")

    for line in f:
        line = line.split("\t")
        for name in [line[1], line[2]] + line[3].split(","):
            name = name.lower()
            if len(name) != 0:
                if name in geo_names:
                    geo_names[name].add((float(line[4]), float(line[5]), int(line[14])))
                else:
                    geo_names[name] = {(float(line[4]), float(line[5]), int(line[14]))}

    conn = sqlite3.connect('../data/geonames.db')
    c = conn.cursor()
    # c.execute("CREATE TABLE GEO (NAME VARCHAR(100) PRIMARY KEY NOT NULL, METADATA VARCHAR(5000) NOT NULL);")
    c.execute("DELETE FROM GEO")
    conn.commit()

    for gn in geo_names:
        c.execute("INSERT INTO GEO VALUES (?, ?)", (gn, str(list(geo_names[gn]))))
    print("Entries saved:", len(geo_names))
    conn.commit()
    conn.close()


def generate_training_data():
    """Prepare Wikipedia training data."""
    conn = sqlite3.connect('../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load('en')
    f = codecs.open("../data/geowiki.txt", "r", encoding="utf-8")
    o = codecs.open("../data/train_wiki.txt", "w", encoding="utf-8")
    lat, lon = u"", u""
    target, string = u"", u""
    skipped = 0

    for line in f:
        if len(line.strip()) == 0:
            continue
        limit = 0
        if line.startswith(u"NEW ARTICLE::"):
            if len(string.strip()) > 0 and len(target) != 0:
                locations = []
                doc = nlp(string)
                for d in doc:
                    if d.text == target[0]:
                        if u" ".join(target) == u" ".join([t.text for t in doc[d.i:d.i + len(target)]]):
                            left = doc[max(0, d.i - 100):d.i]
                            right = doc[d.i + len(target): d.i + len(target) + 100]
                            l, r = [], []
                            location = u""
                            for (out_list, in_list) in [(l, left), (r, right)]:
                                for item in in_list:
                                    if item.ent_type_ in ["GPE", "FACILITY", "LOC"]:
                                        if item.ent_iob_ == "B" and item.text == "the":
                                            out_list.append(u"0.0")
                                        else:
                                            location += item.text + u" "
                                            out_list.append(u"0.0")
                                    elif item.ent_type_ in ["PERSON", "DATE", "TIME", "PERCENT", "MONEY"
                                                            "QUANTITY", "CARDINAL", "ORDINAL"]:
                                        out_list.append(u"0.0")
                                    elif item.is_punct:
                                        out_list.append(u"0.0")
                                    elif item.is_digit or item.like_num:
                                        out_list.append(u"0.0")
                                    elif item.like_email:
                                        out_list.append(u"0.0")
                                    elif item.like_url:
                                        out_list.append(u"0.0")
                                    elif item.is_stop:
                                        out_list.append(u"0.0")
                                    else:
                                        out_list.append(item.text)
                                    if location.strip() != u"" and item.ent_type == 0:
                                        if location.strip() != u" ".join(target):
                                            locations.append(location.strip())
                                        location = u""
                            for i in range(len(locations)):
                                locations[i] = get_coordinates(c, locations[i], pop_only=True)
                            target_grid = get_coordinates(c, u" ".join(target), pop_only=True)
                            if len(target_grid) == 0:
                                skipped += 1
                                break
                            entities_grid = merge_lists(locations)
                            locations = []
                            o.write(lat + u"\t" + lon + u"\t" + str(l) + u"\t" + str(r) + u"\t")
                            o.write(str(target_grid) + u"\t" + str(entities_grid) + u"\t" + u" ".join(target) + u"\t" +
                            u" ".join([s.text for s in left]).strip() + u" ".join([s.text for s in right]).strip() + u"\n")
                            limit += 1
                            if limit > 4:
                                break
            line = line.strip().split("\t")
            if u"(" in line[1]:
                line[1] = line[1].split(u"(")[0].strip()
            if line[1].strip().startswith(u"Geography of "):
                target = line[1].replace(u"Geography of ", u"").split()
            elif u"," in line[1]:
                target = line[1].split(u",")[0].strip().split()
            else:
                target = line[1].split()
            lat = line[2]
            lon = line[3]
            string = ""
            print(u"Processed", limit, u"Skipped:", skipped, u"Name:", u" ".join(target))
        else:
            string += line
    o.close()


def generate_evaluation_data(corpus, gold=False):
    """Prepare WikToR and LGL data. Only the subsets i.e. (2202 WIKTOR, 787 LGL)"""
    conn = sqlite3.connect('../data/geonames.db')
    c = conn.cursor()
    gold = "_gold" if gold else ""
    nlp = spacy.load('en')
    directory = "../data/" + corpus + "/"
    o = codecs.open("./data/eval_" + corpus + gold + ".txt", "w", encoding="utf-8")
    line_no = 0 if corpus == "lgl" else -1

    for line in codecs.open("./data/" + corpus + gold + ".txt", "r", encoding="utf-8"):
        line_no += 1
        if len(line.strip()) == 0:
            continue
        for toponym in line.split("||")[:-1]:
            captured = False
            doc = nlp(codecs.open(directory + str(line_no), "r", encoding="utf-8").read())
            toponym = toponym.split(",,")
            # if toponym[1] == u"Phila.":
            #     print("YEP")
            target = toponym[1].split()
            ent_length = len(u" ".join(target))
            lat, lon = toponym[2], toponym[3]
            start, end = int(toponym[4]), int(toponym[5])
            for d in doc:
                if d.text == target[0]:
                    if u" ".join(target) == u" ".join([t.text for t in doc[d.i:d.i + len(target)]]):
                        locations = []
                        if abs(d.idx - start) > 2 or abs(d.idx + ent_length - end) > 2:
                            continue
                        captured = True
                        left = doc[max(0, d.i - 100):d.i]
                        right = doc[d.i + len(target): d.i + len(target) + 100]
                        l, r = [], []
                        location = u""
                        for (out_list, in_list) in [(l, left), (r, right)]:
                            for item in in_list:
                                if item.ent_type_ in ["GPE", "FACILITY", "LOC", "FAC"]:
                                    if item.ent_iob_ == "B" and item.text == "the":
                                        out_list.append(u"0.0")
                                    else:
                                        location += item.text + u" "
                                        out_list.append(u"0.0")
                                elif item.ent_type_ in ["PERSON", "DATE", "TIME", "PERCENT", "MONEY"
                                                        "QUANTITY", "CARDINAL", "ORDINAL"]:
                                    out_list.append(u"0.0")
                                elif item.is_punct:
                                    out_list.append(u"0.0")
                                elif item.is_digit or item.like_num:
                                    out_list.append(u"0.0")
                                elif item.like_email:
                                    out_list.append(u"0.0")
                                elif item.like_url:
                                    out_list.append(u"0.0")
                                elif item.is_stop:
                                    out_list.append(u"0.0")
                                else:
                                    out_list.append(item.text)
                                if location.strip() != u"" and item.ent_type == 0:
                                    if location.strip() != u" ".join(target):
                                        locations.append(location.strip())
                                    location = u""
                        for i in range(len(locations)):
                            locations[i] = get_coordinates(c, locations[i], pop_only=True)
                        db_entry = toponym[0] if corpus == "lgl" else toponym[1]
                        target_grid = get_coordinates(c, db_entry, pop_only=True)
                        if len(target_grid) == 0:
                            raise Exception(u"No entry in the database!", target)
                        entities_grid = merge_lists(locations)
                        o.write(lat + u"\t" + lon + u"\t" + str(l) + u"\t" + str(r) + u"\t")
                        o.write(str(target_grid) + u"\t" + str(entities_grid) + u"\t" + u" ".join(target) + u"\t" +
                        u" ".join([s.text for s in left]).strip() + u" ".join([s.text for s in right]).strip() + u"\n")
            if not captured:
                print line_no, line
    o.close()


def visualise_2D_grid(x, label):
    """"""
    x = x * 255
    cmap2 = colors.LinearSegmentedColormap.from_list('my_colormap', ['black', 'white', 'red'], 256)
    img2 = pyplot.imshow(np.log(x + 1), interpolation='nearest', cmap=cmap2)
    pyplot.colorbar(img2, cmap=cmap2)
    # plt.imshow(np.log(x + 1), cmap='gray', interpolation='nearest', vmin=0, vmax=np.log(255))
    plt.title(label)
    plt.show()


def generate_vocabulary():
    """Prepare the vocabulary for NN training."""
    vocabulary = {u"<unknown>", u"0.0"}
    temp = []
    training_file = codecs.open("../data/train_wiki.txt", "r", encoding="utf-8")
    for line in training_file:
        line = line.strip().split("\t")
        temp.extend(eval(line[2].lower()))
        temp.extend(eval(line[3].lower()))

    c = Counter(temp)
    for item in c:
        if c[item] > 4:
            vocabulary.add(item)
    cPickle.dump(vocabulary, open("data/vocabulary.pkl", "w"))
    print(u"Vocabulary Size:", len(vocabulary))


def generate_arrays_from_file(path, word_to_index, input_length, batch_size=64, train=True, oneDim=True):
    """"""
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        X_L, X_R, X_E, X_T, Y = [], [], [], [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            Y.append(construct_1D_grid([(float(line[0]), float(line[1]), 0)], use_pop=False))
            X_L.append(pad_list(input_length, eval(line[2].lower()), from_left=True))
            X_R.append(pad_list(input_length, eval(line[3].lower()), from_left=False))
            if oneDim:
                X_T.append(construct_1D_grid(eval(line[4]), use_pop=True))
            else:
                X_T.append(np.array([construct_2D_grid(eval(line[4]), use_pop=True)]))
            if oneDim:
                X_E.append(construct_1D_grid(eval(line[5]), use_pop=False))
            else:
                X_E.append(np.array([construct_2D_grid(eval(line[5]), use_pop=False)]))
            if counter % batch_size == 0:
                for x_l, x_r in zip(X_L, X_R):
                    for i, w in enumerate(x_l):
                        if w in word_to_index:
                            x_l[i] = word_to_index[w]
                        else:
                            x_l[i] = word_to_index[u"<unknown>"]
                    for i, w in enumerate(x_r):
                        if w in word_to_index:
                            x_r[i] = word_to_index[w]
                        else:
                            x_r[i] = word_to_index[u"<unknown>"]
                if train:
                    yield ([np.asarray(X_L), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)], np.asarray(Y))
                else:
                    yield ([np.asarray(X_L), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)])
                X_L, X_R, X_E, X_T, Y = [], [], [], [], []
        if len(Y) > 0:
            # This block is only ever entered at the end to yield the final few samples. (< batch_size)
            for x_l, x_r in zip(X_L, X_R):
                for i, w in enumerate(x_l):
                    if w in word_to_index:
                        x_l[i] = word_to_index[w]
                    else:
                        x_l[i] = word_to_index[u"<unknown>"]
                for i, w in enumerate(x_r):
                    if w in word_to_index:
                        x_r[i] = word_to_index[w]
                    else:
                        x_r[i] = word_to_index[u"<unknown>"]
            if train:
                yield ([np.asarray(X_L), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)], np.asarray(Y))
            else:
                yield ([np.asarray(X_L), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)])


def generate_strings_from_file(path):
    """Returns Y, NAME and CONTEXT"""
    while True:
        for line in codecs.open(path, "r", encoding="utf-8"):
            line = line.strip().split("\t")
            yield ((float(line[0]), float(line[1])), line[6], line[7])


def get_non_zero_entries(a_list):
    """"""
    count = 0
    for col in a_list:
        for row in col:
            if row > 0.0:
                count += 1
    return count

# ----------------------------------------------INVOKE METHODS HERE----------------------------------------------------

# print(list(construct_1D_grid([(86, -179.98333, 10), (86, -174.98333, 0)], use_pop=True)))
# print(list(construct_1D_grid([(90, -180, 0), (90, -170, 1000)], use_pop=True)))
# generate_training_data()
# generate_evaluation_data(corpus="wiki", gold=True)
# index = coord_to_index((-6.43, -172.32), True)
# print(index, index_to_coord(index))
# generate_vocabulary()
# for word in generate_names_from_file("data/eval_lgl.txt"):
#     print word.strip()
# print(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"SaintPierre", pop_only=False))

# conn = sqlite3.connect('../data/geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"saintesprit",
# str(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Saint-Esprit", pop_only=False))))
# conn.commit()

# from geopy.geocoders import geonames
# g = geonames.GeoNames(username='milangritta')
# g = g.geocode(u"Las Vegas", exactly_one=False)

# conn = sqlite3.connect('../data/geonames.db')
# print(len(eval(get_coordinates(conn.cursor(), u"Las Vegas"))))
# print len(g)
# populate_geosql()

# for line in codecs.open("data/eval_wiki.txt", "r", encoding="utf-8"):
#     line = line.strip().split("\t")
#     print line[0], line[1]
#     x = construct_2D_grid(eval(line[4]), use_pop=True)
#     print(get_non_zero_entries(x))
#     visualise_2D_grid(x, line[6] + u" target.")
#     x = construct_2D_grid([(float(line[0]), float(line[1]), 0)], use_pop=False)
#     print(get_non_zero_entries(x))
#     visualise_2D_grid(x, line[6] + u" label.")
#     x = construct_2D_grid(eval(line[5]), use_pop=False)
#     print(get_non_zero_entries(x))
#     visualise_2D_grid(x, line[6] + u" entities.")

# from wikipedia import wikipedia
# search = wikipedia.search(u"N.C.", results=30)
# for s in search:
#     print s


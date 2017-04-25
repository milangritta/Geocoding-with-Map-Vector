# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
# import matplotlib.pyplot as plt
import spacy
import numpy
import sqlite3


def print_stats(accuracy):
    """"""
    accuracy = numpy.log(numpy.array(accuracy) + 1)
    k = numpy.log(161)  # This is the k in accuracy@k metric (see my Survey Paper for details)
    print u"Accuracy to " + str(k) + u" km: ", sum([1.0 for dist in accuracy if dist < k]) / len(accuracy)
    print u"AUC = ", numpy.trapz(accuracy) / (numpy.log(20039) * (len(accuracy) - 1))  # Trapezoidal rule.


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
    grid_size = 5
    latitude = float(coordinates[0]) - 90 if float(coordinates[0]) != -90 else -179.99
    longitude = float(coordinates[1]) + 180 if float(coordinates[1]) != 180 else 359.99
    if longitude < 0:
        longitude = -longitude
    if latitude < 0:
        latitude = -latitude
    if not get_xy:
        return latitude, longitude
    x = (360 / grid_size) * (int(latitude) / grid_size)
    y = int(longitude) / grid_size
    return x + y if 0 <= x + y <= 2592 else Exception(u"Shock horror!!")


def index_to_coord(index):
    """"""
    x = int(index / 72)
    y = index % 72
    if x > 18:
        x = -(x - 18) * 5
    else:
        x = (18 - x) * 5
    if y < 36:
        y = -(36 - y) * 5
    else:
        y = (y - 36) * 5
    return x, y


def get_coordinates(con, loc_name):
    """"""
    result = con.execute(u"SELECT * FROM GEO WHERE NAME = ?", (loc_name, )).fetchone()
    return result[1] if result else '[]'


def construct_1D_grid(a_list, use_pop):
    """"""
    g = numpy.zeros(2592)
    for s in a_list:
        if use_pop:
            g[coord_to_index((s[0], s[1]), True)] += numpy.log(numpy.e + s[2])
        else:
            g[coord_to_index((s[0], s[1]), True)] += 1
    return g / max(g) if max(g) > 0.0 else g


def construct_2D_grid(a_list, use_pop):
    """"""
    g = numpy.zeros((36, 72))
    for s in a_list:
        x, y = coord_to_index((s[0], s[1]), False)
        x = int(x / 5)
        y = int(y / 5)
        if use_pop:
            g[x][y] += numpy.log(numpy.e + s[2])
        else:
            g[x][y] += 1
    return g / numpy.amax(g) if numpy.amax(g) > 0.0 else g


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
    entity, string = u"", u""
    skipped = 0

    for line in f:
        if len(line.strip()) == 0:
            continue
        limit = 0
        if line.startswith(u"NEW ARTICLE::"):
            if len(string.strip()) > 0 and len(entity) != 0:
                locations = []
                doc = nlp(string)
                for d in doc:
                    if d.text == entity[0]:
                        if u" ".join(entity) == u" ".join([t.text for t in doc[d.i:d.i + len(entity)]]):
                            left = doc[max(0, d.i - 50):d.i]
                            right = doc[d.i + len(entity): d.i + len(entity) + 50]
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
                                    if location != u"" and item.ent_type == 0:
                                        locations.append(location.strip())
                                        location = u""
                            for i in range(len(locations)):
                                locations[i] = eval(get_coordinates(c, locations[i]))
                            ent_grid = get_coordinates(c, u" ".join(entity))
                            if len(eval(ent_grid)) == 0:
                                skipped += 1
                                break
                            loc_grid = merge_lists(locations)
                            locations = []
                            o.write(lat + u"\t" + lon + u"\t" + str(l) + u"\t" + str(r) + u"\t")
                            o.write(ent_grid + u"\t" + str(loc_grid) + u"\t" + u" ".join(entity) + u"\n")
                            limit += 1
                            if limit > 4:
                                break
            line = line.strip().split("\t")
            if u"(" in line[1]:
                line[1] = line[1].split(u"(")[0].strip()
            if line[1].strip().startswith(u"Geography of "):
                entity = line[1].replace(u"Geography of ", u"").split()
            elif u"," in line[1]:
                entity = line[1].split(u",")[0].strip().split()
            else:
                entity = line[1].split()
            lat = line[2]
            lon = line[3]
            string = ""
            print(u"Processed", limit, u"Skipped:", skipped, u"Name:", u" ".join(entity))
        else:
            string += line
    o.close()


def generate_evaluation_data():
    """Prepare WikToR and LGL data. Only the subsets i.e. (2202 WIKTOR, 787 LGL)"""
    conn = sqlite3.connect('../data/geonames.db')
    c = conn.cursor()
    corpus = "wiki"
    nlp = spacy.load('en')
    directory = "/Users/milangritta/PycharmProjects/Research/" + corpus + "/"
    o = codecs.open("./data/eval_" + corpus + ".txt", "w", encoding="utf-8")
    line_no = 0 if corpus == "lgl" else -1

    for line in codecs.open("./data/" + corpus + ".txt", "r", encoding="utf-8"):
        line_no += 1
        if len(line.strip()) == 0:
            continue
        for toponym in line.split("||")[:-1]:
            captured = False
            doc = nlp(codecs.open(directory + str(line_no), "r", encoding="utf-8").read())
            toponym = toponym.split(",,")
            entity = toponym[1].split()
            ent_length = len(entity)
            lat, lon = toponym[2], toponym[3]
            start, end = int(toponym[4]), int(toponym[5])
            for d in doc:
                if d.text == entity[0]:
                    if u" ".join(entity) == u" ".join([t.text for t in doc[d.i:d.i + len(entity)]]):
                        locations = []
                        if d.idx != start and d.idx + ent_length != end:
                            continue
                        captured = True
                        left = doc[max(0, d.i - 50):d.i]
                        right = doc[d.i + len(entity): d.i + len(entity) + 50]
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
                                if location != u"" and item.ent_type == 0:
                                    locations.append(location.strip())
                                    location = u""
                        for i in range(len(locations)):
                            locations[i] = eval(get_coordinates(c, locations[i]))
                        ent_grid = get_coordinates(c, u" ".join(entity))
                        # if len(eval(ent_grid)) == 0:
                        #     break
                        loc_grid = merge_lists(locations)
                        o.write(lat + u"\t" + lon + u"\t" + str(l) + u"\t" + str(r) + u"\t")
                        o.write(ent_grid + u"\t" + str(loc_grid) + u"\t" + u" ".join(entity) + u"\n")
            if not captured:
                print line
    o.close()


# def visualise_2D_grid():
#     """"""
#     x = construct_2D_grid(eval(line[5])) * 255
#     plt.imshow(numpy.log(x + 1), cmap='gray', interpolation='nearest', vmin=0, vmax=numpy.log(255))
#     plt.title(line[6])
#     plt.show()


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
        if c[item] > 1:
            vocabulary.add(item)
    cPickle.dump(vocabulary, open("data/vocabulary.pkl", "w"))
    print(u"Vocabulary Size:", len(vocabulary))

# ----------------------------------------------INVOKE METHODS HERE----------------------------------------------------

# print(coord_to_grid(('86', '-179.98333')))
# print(coord_to_grid((-90, 180)))
# generate_training_data()
# generate_evaluation_data()
# index = coord_to_index((-6.43, -172.32), True)
# print(index, index_to_coord(index))
# generate_vocabulary()

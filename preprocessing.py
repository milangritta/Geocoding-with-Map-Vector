# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
from scipy import ndimage
import matplotlib.pyplot as plt
import spacy
import numpy as np
import sqlite3
from geopy.distance import great_circle
from matplotlib import pyplot, colors
from scipy.spatial.distance import euclidean


# -------- GLOBAL CONSTANTS AND VARIABLES -------- #
BATCH_SIZE = 64
CONTEXT_LENGTH = 200
UNKNOWN = u"<unknown>"
PADDING = u"0"
EMB_DIM = 50
TARGET_LENGTH = 15
FILTER_1x1 = cPickle.load(open(u"data/1x1_filter.pkl"))    # We need these filters
FILTER_2x2 = cPickle.load(open(u"data/2x2_filter.pkl"))    # and the reverse ones
REVERSE_1x1 = cPickle.load(open(u"data/1x1_reverse.pkl"))  # to handle the used and
REVERSE_2x2 = cPickle.load(open(u"data/2x2_reverse.pkl"))  # unused loc2vec polygons.
# -------- GLOBAL CONSTANTS AND VARIABLES -------- #


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
            a_list = [PADDING] + a_list
        else:
            a_list += [PADDING]
    return a_list


def coord_to_index(coordinates, polygon_size):
    """"""
    latitude = float(coordinates[0]) - 90 if float(coordinates[0]) != -90 else -179.99  # The two edge cases must
    longitude = float(coordinates[1]) + 180 if float(coordinates[1]) != 180 else 359.99  # get handled differently!
    if longitude < 0:
        longitude = -longitude
    if latitude < 0:
        latitude = -latitude
    x = int(360 / polygon_size) * int(latitude / polygon_size)
    y = int(longitude / polygon_size)
    return x + y if 0 <= x + y <= int(360 / polygon_size) * int(180 / polygon_size) else Exception(u"Shock horror!!")


def index_to_coord(index, polygon_size):
    """"""
    x = int(index / (360 / polygon_size))
    y = index % int(360 / polygon_size)
    if x > int(90 / polygon_size):
        x = -int((x - (90 / polygon_size)) * polygon_size)
    else:
        x = int(((90 / polygon_size) - x) * polygon_size)
    if y < int(180 / polygon_size):
        y = -int(((180 / polygon_size) - y) * polygon_size)
    else:
        y = int((y - (180 / polygon_size)) * polygon_size)
    return x, y


def get_coordinates(con, loc_name):
    """"""
    result = con.execute(u"SELECT METADATA FROM GEO WHERE NAME = ?", (loc_name.lower(),)).fetchone()
    if result:
        result = eval(result[0])  # Do not remove the sorting, the function below assumes sorted results!
        return sorted(result, key=lambda (a, b, c, d): c, reverse=True)
    else:
        return []


def construct_loc2vec(a_list, polygon_size, filter_type):
    """"""
    loc2vec = g = np.zeros(len(filter_type),)
    if len(a_list) == 0:
        return loc2vec
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        loc2vec[filter_type[index]] += float(max(s[2], 1)) / max_pop
    return loc2vec / loc2vec.max() if loc2vec.max() > 0.0 else loc2vec


def assemble_features(target, near, far, polygon_size, filter_type):
    """"""
    target = construct_loc2vec(target, polygon_size, filter_type)
    near = construct_loc2vec(near, polygon_size, filter_type)
    far = construct_loc2vec(far, polygon_size, filter_type)
    l2v = np.add(np.add(near, far), target)
    return l2v / l2v.max()


def merge_lists(grids):
    """"""
    out = []
    for g in grids:
        out.extend(g)
    return out


def populate_sql():
    """Create and populate the sqlite database with GeoNames data"""
    geo_names = {}
    p_map = {"PPLC": 100000, "PCLI": 100000, "PCL": 100000, "PCLS": 10000, "PCLF": 10000, "CONT": 100000, "RGN": 100000}

    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        feat_code = line[7]
        class_code = line[6]
        pop = int(line[14])
        for name in [line[1], line[2]] + line[3].split(","):
            name = name.lower()
            if len(name) != 0:
                if name in geo_names:
                    already_have_entry = False
                    for item in geo_names[name]:
                        if great_circle((float(line[4]), float(line[5])), (item[0], item[1])).km < 100:
                            if item[2] >= pop:
                                already_have_entry = True
                    if not already_have_entry:
                        pop = get_population(class_code, feat_code, p_map, pop)
                        geo_names[name].add((float(line[4]), float(line[5]), pop, feat_code))
                else:
                    pop = get_population(class_code, feat_code, p_map, pop)
                    geo_names[name] = {(float(line[4]), float(line[5]), pop, feat_code)}

    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    # c.execute("CREATE TABLE GEO (NAME VARCHAR(100) PRIMARY KEY NOT NULL, METADATA VARCHAR(5000) NOT NULL);")
    c.execute(u"DELETE FROM GEO")  # alternatively, delete the database file.
    conn.commit()

    for gn in geo_names:
        c.execute(u"INSERT INTO GEO VALUES (?, ?)", (gn, str(list(geo_names[gn]))))
    print(u"Entries saved:", len(geo_names))
    conn.commit()
    conn.close()


def get_population(class_code, feat_code, p_map, pop):
    """"""
    if pop == 0 and class_code in ["A", "P", "L"]:
        pop = p_map.get(feat_code, 0)
    return pop


# def remove_redundant_polygons(loc2vec, polygon_size):
#     """"""
#     filter_type = FILTER_1x1 if polygon_size == 1 else FILTER_2x2
#     g = np.zeros(len(filter_type),)
#     for index in filter_type:
#         g[index] = loc2vec[filter_type[index]]
#     return g


def generate_training_data():
    """Prepare Wikipedia training data."""
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load(u'en')
    inp = codecs.open(u"../data/geowiki.txt", u"r", encoding=u"utf-8")
    o = codecs.open(u"../data/train_wiki.txt", u"w", encoding=u"utf-8")
    lat, lon = u"", u""
    target, string = u"", u""
    skipped = 0

    for line in inp:
        if len(line.strip()) == 0:
            continue
        limit = 0
        if line.startswith(u"NEW ARTICLE::"):
            if len(string.strip()) > 0 and len(target) != 0:
                locations_near, locations_far = [], []
                doc = nlp(string)
                for d in doc:
                    if d.text == target[0]:
                        if u" ".join(target) == u" ".join([t.text for t in doc[d.i:d.i + len(target)]]):
                            near_inp = [x for x in doc[max(0, d.i - CONTEXT_LENGTH / 2):d.i]] + \
                                       [x for x in doc[d.i + len(target): d.i + len(target) + CONTEXT_LENGTH / 2]]
                            far_inp = [x for x in doc[max(0, d.i - CONTEXT_LENGTH):max(0, d.i - CONTEXT_LENGTH / 2)]] + \
                                      [x for x in doc[d.i + len(target) + CONTEXT_LENGTH / 2: d.i + len(target) + CONTEXT_LENGTH]]
                            near_out, far_out = [], []
                            location = u""
                            for (out_list, in_list, is_near) in [(near_out, near_inp, True), (far_out, far_inp, False)]:
                                for index, item in enumerate(in_list):
                                    if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
                                        if item.ent_iob_ == u"B" and item.text.lower() == u"the":
                                            out_list.append(item.text.lower())
                                        else:
                                            location += item.text + u" "
                                            out_list.append(u"**LOC**" + item.text.lower())
                                    elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY",
                                                            u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
                                        out_list.append(PADDING)
                                    elif item.is_punct:
                                        out_list.append(PADDING)
                                    elif item.is_digit or item.like_num:
                                        out_list.append(PADDING)
                                    elif item.like_email:
                                        out_list.append(PADDING)
                                    elif item.like_url:
                                        out_list.append(PADDING)
                                    elif item.is_stop:
                                        out_list.append(PADDING)
                                    else:
                                        out_list.append(item.lemma_)
                                    if location.strip() != u"" and (item.ent_type == 0 or index == len(in_list) - 1):
                                        location = location.strip()
                                        coords = get_coordinates(c, location)
                                        if len(coords) > 0 and location != u" ".join(target):
                                            if is_near:
                                                locations_near.append(coords)
                                            else:
                                                locations_far.append(coords)
                                        else:
                                            offset = 1 if index == len(in_list) - 1 else 0
                                            for i in range(index - len(location.split()), index):
                                                out_list[i + offset] = in_list[i + offset].lemma_ \
                                                if in_list[i + offset].is_alpha and location != u" ".join(target) else PADDING
                                        location = u""
                            target_grid = get_coordinates(c, u" ".join(target))
                            if len(target_grid) == 0:
                                skipped += 1
                                break
                            entities_near = merge_lists(locations_near)
                            entities_far = merge_lists(locations_far)
                            locations_near, locations_far = [], []
                            o.write(lat + u"\t" + lon + u"\t" + str(near_out) + u"\t" + str(far_out) + u"\t")
                            o.write(str(target_grid) + u"\t" + str([t.lower() for t in target][:TARGET_LENGTH]))
                            o.write(u"\t" + str(entities_near) + u"\t" + str(entities_far) + u"\n")
                            limit += 1
                            if limit > 29:
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


def generate_evaluation_data(corpus, file_name):
    """"""
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load(u'en')
    directory = u"../data/" + corpus + u"/"
    o = codecs.open(u"data/eval_" + corpus + file_name + u".txt", u"w", encoding=u"utf-8")
    line_no = 0 if corpus == u"lgl" else -1

    for line in codecs.open(u"data/" + corpus + file_name + u".txt", u"r", encoding=u"utf-8"):
        line_no += 1
        if len(line.strip()) == 0:
            continue
        for toponym in line.split(u"||")[:-1]:
            captured = False
            doc = nlp(codecs.open(directory + str(line_no), u"r", encoding=u"utf-8").read())
            locations_near, locations_far = [], []
            toponym = toponym.split(u",,")
            target = [t.text for t in nlp(toponym[1])]
            ent_length = len(u" ".join(target))
            lat, lon = toponym[2], toponym[3]
            start, end = int(toponym[4]), int(toponym[5])
            for d in doc:
                if d.text == target[0]:
                    if u" ".join(target) == u" ".join([t.text for t in doc[d.i:d.i + len(target)]]):
                        if abs(d.idx - start) > 4 or abs(d.idx + ent_length - end) > 4:
                            continue
                        captured = True
                        near_inp = [x for x in doc[max(0, d.i - CONTEXT_LENGTH / 2):d.i]] + \
                                   [x for x in doc[d.i + len(target): d.i + len(target) + CONTEXT_LENGTH / 2]]
                        far_inp = [x for x in doc[max(0, d.i - CONTEXT_LENGTH):max(0, d.i - CONTEXT_LENGTH / 2)]] + \
                                  [x for x in doc[d.i + len(target) + CONTEXT_LENGTH / 2: d.i + len(target) + CONTEXT_LENGTH]]
                        near_out, far_out = [], []
                        location = u""
                        for (out_list, in_list, is_near) in [(near_out, near_inp, True), (far_out, far_inp, False)]:
                            for index, item in enumerate(in_list):
                                if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
                                    if item.ent_iob_ == u"B" and item.text.lower() == u"the":
                                        out_list.append(item.text.lower())
                                    else:
                                        location += item.text + u" "
                                        out_list.append(u"**LOC**" + item.text.lower())
                                elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY",
                                                        u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
                                    out_list.append(PADDING)
                                elif item.is_punct:
                                    out_list.append(PADDING)
                                elif item.is_digit or item.like_num:
                                    out_list.append(PADDING)
                                elif item.like_email:
                                    out_list.append(PADDING)
                                elif item.like_url:
                                    out_list.append(PADDING)
                                elif item.is_stop:
                                    out_list.append(PADDING)
                                else:
                                    out_list.append(item.lemma_)
                                if location.strip() != u"" and (item.ent_type == 0 or index == len(in_list) - 1):
                                    location = location.strip()
                                    coords = get_coordinates(c, location)
                                    if len(coords) > 0 and location != u" ".join(target):
                                        if is_near:
                                            locations_near.append(coords)
                                        else:
                                            locations_far.append(coords)
                                    else:
                                        offset = 1 if index == len(in_list) - 1 else 0
                                        for i in range(index - len(location.split()), index):
                                            out_list[i + offset] = in_list[i + offset].lemma_ \
                                                if in_list[i + offset].is_alpha and location != u" ".join(target) else PADDING
                                    location = u""

                        lookup = toponym[0] if corpus != u"wiki" else toponym[1]
                        target_grid = get_coordinates(c, lookup)
                        if len(target_grid) == 0:
                            raise Exception(u"No entry in the database!", lookup)
                        entities_near = merge_lists(locations_near)
                        entities_far = merge_lists(locations_far)
                        locations_near, locations_far = [], []
                        o.write(lat + u"\t" + lon + u"\t" + str(near_out) + u"\t" + str(far_out) + u"\t")
                        o.write(str(target_grid) + u"\t" + str([t.lower() for t in lookup.split()][:TARGET_LENGTH]))
                        o.write(u"\t" + str(entities_near) + u"\t" + str(entities_far) + u"\n")
            if not captured:
                print line_no, line, target, start, end
    o.close()


def visualise_2D_grid(x, title, log=False):
    """"""
    if log:
        x = np.log10(x)
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap', ['yellow', 'orange', 'red', 'black'])
    cmap.set_bad(color='lightblue')
    img = pyplot.imshow(x, cmap=cmap, interpolation='nearest')
    pyplot.colorbar(img, cmap=cmap)
    plt.title(title)
    # plt.savefig(u"images/" + title + u".png", dpi=200)
    plt.show()


def apply_smoothing(loc2vec, polygon_size, sigma):
    """"""
    loc2vec = np.reshape(loc2vec, newshape=((180 / polygon_size), (360 / polygon_size)))
    # for row in loc2vec:
    #     print list(row)
    loc2vec = ndimage.filters.gaussian_filter(loc2vec, sigma)
    # for row in loc2vec:
    #     print list(row)
    return loc2vec.ravel()


def generate_vocabulary():
    """Prepare the vocabulary(ies) for training."""
    vocab_words, vocab_locations = {UNKNOWN, PADDING}, {UNKNOWN, PADDING}
    words, locations = [], []
    for f in [u"../data/train_wiki.txt"]:
        training_file = codecs.open(f, u"r", encoding=u"utf-8")
        for line in training_file:
            line = line.strip().split("\t")
            words.extend([w for w in eval(line[2]) if u"**LOC**" not in w])  # NEAR WORDS
            words.extend([w for w in eval(line[3]) if u"**LOC**" not in w])  # FAR WORDS
            locations.extend([w for w in eval(line[2]) if u"**LOC**" in w])  # NEAR ENTITIES
            locations.extend([w for w in eval(line[3]) if u"**LOC**" in w])  # FAR ENTITIES

    words = Counter(words)
    for word in words:
        if words[word] > 9:
            vocab_words.add(word)
    print(u"Words saved:", len(vocab_words))

    locations = Counter(locations)
    for location in locations:
        if locations[location] > 1:
            vocab_locations.add(location.replace(u"**LOC**", u""))
    print(u"Locations saved:", len(vocab_locations))

    vocabulary = vocab_words.union(vocab_locations)
    word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])
    cPickle.dump(word_to_index, open(u"data/w2i.pkl", "w"))


def generate_arrays_from_file(path, w2i, train=True):
    """"""
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        context_words, entities_strings, labels = [], [], []
        loc2vec, target_string = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_loc2vec([(float(line[0]), float(line[1]), 0)], 2, FILTER_2x2))

            near = [w if u"**LOC**" not in w else PADDING for w in eval(line[2])]
            far = [w if u"**LOC**" not in w else PADDING for w in eval(line[3])]
            context_words.append(pad_list(CONTEXT_LENGTH, None, from_left=True))

            near = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else PADDING for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else PADDING for w in eval(line[3])]
            entities_strings.append(pad_list(CONTEXT_LENGTH, None, from_left=True))

            polygon_size = 2

            loc2vec.append(assemble_features(eval(line[4]), eval(line[6]), eval(line[7]), polygon_size, FILTER_1x1))

            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), from_left=True))

            if counter % BATCH_SIZE == 0:
                for collection in [context_words, entities_strings, target_string]:
                    for x in collection:
                        for i, w in enumerate(x):
                            if w in w2i:
                                x[i] = w2i[w]
                            else:
                                x[i] = w2i[UNKNOWN]
                if train:
                    yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                            np.asarray(entities_strings), np.asarray(loc2vec), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                            np.asarray(entities_strings), np.asarray(loc2vec), np.asarray(target_string)])

                context_words, entities_strings, labels = [], [], []
                loc2vec, target_string = [], []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for collection in [context_words, entities_strings, target_string]:
                for x in collection:
                    for i, w in enumerate(x):
                        if w in w2i:
                            x[i] = w2i[w]
                        else:
                            x[i] = w2i[UNKNOWN]
            if train:
                yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                        np.asarray(entities_strings), np.asarray(loc2vec), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                        np.asarray(entities_strings), np.asarray(loc2vec), np.asarray(target_string)])


def generate_arrays_from_file_lstm(path, w2i, train=True):
    """"""
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        context_words_left, context_words_right, entities_strings_left, entities_strings_right = [], [], [], []
        target_string, labels = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_loc2vec([(float(line[0]), float(line[1]), 0)], 2))

            near = [w if u"**LOC**" not in w else PADDING for w in eval(line[2])]
            far = [w if u"**LOC**" not in w else PADDING for w in eval(line[3])]
            context_words_left.append(pad_list(CONTEXT_LENGTH, far[:CONTEXT_LENGTH / 2]
                                               + near[:CONTEXT_LENGTH / 2], from_left=True))
            context_words_right.append(pad_list(CONTEXT_LENGTH, near[CONTEXT_LENGTH / 2:]
                                               + far[CONTEXT_LENGTH / 2:], from_left=False))

            near = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else PADDING for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else PADDING for w in eval(line[3])]
            entities_strings_left.append(pad_list(CONTEXT_LENGTH, far[:CONTEXT_LENGTH / 2]
                                               + near[:CONTEXT_LENGTH / 2], from_left=True))
            entities_strings_right.append(pad_list(CONTEXT_LENGTH, near[CONTEXT_LENGTH / 2:]
                                               + far[CONTEXT_LENGTH / 2:], from_left=False))

            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), from_left=True))

            if counter % BATCH_SIZE == 0:
                for collection in [context_words_left, context_words_right, entities_strings_left, entities_strings_right, target_string]:
                    for x in collection:
                        for i, w in enumerate(x):
                            if w in w2i:
                                x[i] = w2i[w]
                            else:
                                x[i] = w2i[UNKNOWN]
                if train:
                    yield ([np.asarray(context_words_left), np.asarray(context_words_right), np.asarray(entities_strings_left),
                            np.asarray(entities_strings_right), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(context_words_left), np.asarray(context_words_right), np.asarray(entities_strings_left),
                            np.asarray(entities_strings_right), np.asarray(target_string)])

                context_words_left, context_words_right, entities_strings_left, entities_strings_right = [], [], [], []
                target_string, labels = [], []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for collection in [context_words_left, context_words_right, entities_strings_left, entities_strings_right, target_string]:
                for x in collection:
                    for i, w in enumerate(x):
                        if w in w2i:
                            x[i] = w2i[w]
                        else:
                            x[i] = w2i[UNKNOWN]
            if train:
                yield ([np.asarray(context_words_left), np.asarray(context_words_right), np.asarray(entities_strings_left),
                        np.asarray(entities_strings_right), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(context_words_left), np.asarray(context_words_right), np.asarray(entities_strings_left),
                        np.asarray(entities_strings_right), np.asarray(target_string)])


def generate_strings_from_file(path):
    """Returns Y, NAME and CONTEXT"""
    while True:
        for line in codecs.open(path, "r", encoding="utf-8"):
            line = line.strip().split("\t")
            context = u" ".join(eval(line[2])) + u"*E*" + u" ".join(eval(line[5])) + u"*E*" + u" ".join(eval(line[3]))
            yield ((float(line[0]), float(line[1])), u" ".join(eval(line[5])).strip(), context)


def compute_embedding_distances(W, dim, polygon_size):
    store = []
    W = np.reshape(W, (int(180 / polygon_size), int(360 / polygon_size), dim))
    for row in W:
        store_col = []
        for column in row:
            col_vector = []
            for r in W:
                for c in r:
                    col_vector.append(euclidean(column, c))
            store_col.append(col_vector)
        store.append(store_col)
    return store


def compute_pixel_similarity(polygon_size):
    distances_p = compute_embedding_distances(cPickle.load(open("data/W.pkl")), 801, polygon_size)

    store = []
    for r in range(int(180 / polygon_size)):
        store_c = []
        for c in range(int(360 / polygon_size)):
            store_c.append((r, c))
        store.append(store_c)

    distances_g = compute_embedding_distances(np.array(store), 2, polygon_size)

    correlations = []
    for p, g in zip(distances_p, distances_g):
        for cp, cg in zip(p, g):
            correlations.append(np.corrcoef(cp, cg))

    cPickle.dump(correlations, open(u"data/correlations.pkl", "w"))


def filter_wiktor():
    wiktor = set()
    for line in codecs.open(u"data/eval_wiki_gold.txt", "r", encoding="utf-8"):
        wiktor.add(line)
    print(len(wiktor))
    for line in codecs.open(u"../data/train_wiki_uniform.txt", "r", encoding="utf-8"):
        if line in wiktor:
            print line


def training_map(polygon_size):
    coordinates = []
    for f in [u"../data/train_wiki_uniform.txt"]:
        training_file = codecs.open(f, "r", encoding="utf-8")
        for line in training_file:
            line = line.strip().split("\t")
            coordinates.append((float(line[0]), float(line[1]), 0))
    c = construct_loc2vec(coordinates, polygon_size, FILTER_1x1)
    c = np.reshape(c, (int(180 / polygon_size), int(360 / polygon_size)))
    visualise_2D_grid(c, u"Training Map", log=True)


def generate_arrays_from_file_loc(path, train=True, looping=True):
    """"""
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        polygon_size = 1
        labels, target_coord = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_loc2vec([(float(line[0]), float(line[1]), 0)], 2, FILTER_2x2))
            # X = apply_smoothing(construct_loc2vec(\
            # eval(line[4]), eval(line[6]), eval(line[7]), polygon_size), polygon_size, sigma=0.4)
            # target_coord.append(X / X.max())
            target_coord.append(assemble_features(eval(line[4]), eval(line[6]), eval(line[7]), polygon_size, FILTER_1x1))

            if counter % BATCH_SIZE == 0:
                if train:
                    yield ([np.asarray(target_coord)], np.asarray(labels))
                else:
                    yield ([np.asarray(target_coord)])

                labels = []
                target_coord = []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            if train:
                yield ([np.asarray(target_coord)], np.asarray(labels))
            else:
                yield ([np.asarray(target_coord)])
        if not looping:
            break


def shrink_loc2vec(polygon_size):
    """Remove polygons that cover the oceans."""
    loc2vec = np.zeros((180 / polygon_size) * (360 / polygon_size),)
    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        lat, lon = float(line[4]), float(line[5])
        index = coord_to_index((lat, lon), polygon_size=polygon_size)
        loc2vec[index] += 1.0
    cPickle.dump(loc2vec, open(u"loc2vec.pkl", "w"))

# --------------------------------------------- INVOKE METHODS HERE ---------------------------------------------------

# training_map()

# visualise_2D_grid(construct_2D_grid(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"washington")), "image")

# print get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"china")

# generate_training_data()

# generate_evaluation_data(corpus="lgl", file_name="")

# generate_vocabulary()

# shrink_loc2vec(2)

# conn = sqlite3.connect('../data/geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"darfur", u"[(13.5, 23.5, 0), (13.0, 25.0, 10000), (44.05135, -94.83804, 106)]"))
# c.execute("DELETE FROM GEO WHERE name = 'darfur'")
# conn.commit()

# populate_sql()

# for line in codecs.open("data/eval_wiki.txt", "r", encoding="utf-8"):
#     line = line.strip().split("\t")
#     x = construct_loc2vec(eval(line[4]), eval(line[6]), eval(line[7]), polygon_size=2)
#     x = np.reshape(x, newshape=((180 / 2), (360 / 2)))
#     visualise_2D_grid(x, " ".join(eval(line[5])))
#     x = apply_smoothing(x, polygon_size=2, sigma=0.4)
#     x = np.reshape(x, newshape=((180 / 2), (360 / 2)))
#     visualise_2D_grid(x, " ".join(eval(line[5])))

# c = Counter(c)
# counts = []
# for key in c.most_common():
#     counts.append(key[1])
# print(len(c)/4462.0)
# y_pos = np.arange(len(counts))
# plt.bar(y_pos, counts, align='center', alpha=0.5)
# plt.ylabel('Counts')
# plt.title('Toponym Counts')
# plt.show()

# counter = 0
# out = codecs.open("data/test.txt", "w", encoding="utf-8")
# for line in codecs.open("data/wiki.txt", "r", encoding="utf-8"):
#     if counter % 3 == 0:
#         out.write(line)
#     counter += 1

# l2v = list(cPickle.load(open(u"data/geonames_1x1.pkl")))
# zeros = dict([(i, v) for i, v in enumerate(l2v) if v > 0])  # isolate the non zero values
# zeros = dict([(i, v) for i, v in enumerate(zeros)])         # replace counts with indices
# zeros = dict([(v, i) for (i, v) in zeros.iteritems()])      # reverse keys and values
# cPickle.dump(zeros, open(u"data/1x1_filter.pkl", "w"))

# filtered = [i for i, v in enumerate(l2v) if v > 0]
# the_rest = [i for i, v in enumerate(l2v) if v == 0]
# poly_size = 1
# dict_rest = dict()
#
# for poly_rest in the_rest:
#     best_index = 100000
#     best_dist = 100000
#     for poly_filtered in filtered:
#         dist = great_circle(index_to_coord(poly_rest, poly_size), index_to_coord(poly_filtered, poly_size)).km
#         if dist < best_dist:
#             best_index = poly_filtered
#             best_dist = dist
#     dict_rest[poly_rest] = best_index
#
# cPickle.dump(dict_rest, open(u"data/1x1_rest.pkl", "w"))

# l2v = np.reshape(l2v, newshape=((180 / 1), (360 / 1)))
# visualise_2D_grid(l2v, "Geonames Database", True)

# plt.plot(range(len(l2v)), np.asarray(sorted(l2v)))
# plt.xlabel(u"Predictions")
# plt.ylabel(u'Error Size')
# plt.title(u"Some Chart")
# plt.savefig(u'test.png', transparent=True)
# plt.show()

# correlations = [x[0][1] for x in cPickle.load(open("data/correlations.pkl"))]
# correlations = [x[0][1] for x in correlations]
# minimum = min(correlations)
# ran = max(correlations) - minimum
# correlations = [x + ran for x in correlations]
# correlations = np.reshape(np.array(correlations), ((180 / GRID_SIZE), (360 / GRID_SIZE)))
# correlations = np.rot90((np.rot90(correlations)))
# visualise_2D_grid(correlations, "GeoPixelSpatialCorrelation")

# print index_to_coord(8177, 2)

# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()
# CODE HERE
# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()

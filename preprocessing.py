# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import numpy as np
import sqlite3
from geopy.distance import great_circle
from matplotlib import pyplot, colors


# -------- GLOBAL CONSTANTS AND VARIABLES -------- #
BATCH_SIZE = 64
CONTEXT_LENGTH = 200
UNKNOWN = u"<unknown>"
EMBEDDING_DIMENSION = 50
TARGET_LENGTH = 15
ENCODING_MAP_1x1 = cPickle.load(open(u"data/1x1_encode_map.pkl"))      # We need these maps
ENCODING_MAP_2x2 = cPickle.load(open(u"data/2x2_encode_map.pkl"))      # and the reverse ones
REVERSE_MAP_1x1 = cPickle.load(open(u"data/1x1_reverse_map.pkl"))      # to handle the used and
REVERSE_MAP_2x2 = cPickle.load(open(u"data/2x2_reverse_map.pkl"))      # unused map2vec polygons.
OUTLIERS_MAP_1x1 = cPickle.load(open(u"data/1x1_outliers_map.pkl"))    # Outliers are redundant polygons that
OUTLIERS_MAP_2x2 = cPickle.load(open(u"data/2x2_outliers_map.pkl"))    # have been removed but must also be handled.
# -------- GLOBAL CONSTANTS AND VARIABLES -------- #


def print_stats(accuracy):
    """
    Prints mean, median, AUC and acc@161km for the list.
    :param accuracy: a list of geocoding errors
    """
    print("==============================================================================================")
    print(u"Median error:", np.median(sorted(accuracy)))
    print(u"Mean error:", np.mean(accuracy))
    accuracy = np.log(np.array(accuracy) + 1)
    k = np.log(161)
    print u"Accuracy to 161 km: ", sum([1.0 for dist in accuracy if dist < k]) / len(accuracy)
    print u"AUC = ", np.trapz(accuracy) / (np.log(20039) * (len(accuracy) - 1))  # Trapezoidal rule.
    print("==============================================================================================")


def pad_list(size, a_list, from_left, padding):
    """
    Pads a given list with any given padding.
    :param size: the final length of the list i.e. pad up to size
    :param a_list: the list to pad
    :param from_left: True to pad from the left, False to pad from the right
    :param padding: whatever you want to use for padding, example "0"
    :return: the padded list
    """
    while len(a_list) < size:
        if from_left:
            a_list = [padding] + a_list
        else:
            a_list += [padding]
    return a_list


def coord_to_index(coordinates, polygon_size):
    """
    Convert coordinates into an array index. Use that to modify map2vec polygon value.
    :param coordinates: (latitude, longitude) to compute
    :param polygon_size: integer size of the polygon? i.e. the resolution of the world
    :return: index pointing into map2vec array
    """
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
    """
    Convert index to coordinates.
    :param index: of the polygon in map2vec array
    :param polygon_size: size of each polygon i.e. resolution of the world
    :return: (latitude, longitude)
    """
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
    """
    Access the database to retrieve coordinates and other data from DB.
    :param con: sqlite3 database cursor i.e. DB connection
    :param loc_name: name of the place
    :return: a list of tuples [(latitude, longitude, population, feature_code), ...]
    """
    result = con.execute(u"SELECT METADATA FROM GEO WHERE NAME = ?", (loc_name.lower(),)).fetchone()
    if result:
        result = eval(result[0])  # Do not remove the sorting, the function below assumes sorted results!
        return sorted(result, key=lambda (a, b, c, d): c, reverse=True)
    else:
        return []


def construct_map2vec(a_list, polygon_size, mapping, outliers):
    """
    Build the map2vec vector representation from a list of location data.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: what's the resolution? size of each polygon in degrees.
    :param mapping: one of the transformation maps 1x1 or 2x2
    :param outliers: the outlier map, 1x1 or 2x2 (must match resolution or mapping above)
    :return: map2vec vector representation
    """
    map2vec = np.zeros(len(mapping), )
    if len(a_list) == 0:
        return map2vec
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        if index in mapping:
            index = mapping[index]
        else:
            index = mapping[outliers[index]]
        map2vec[index] += float(max(s[2], 1)) / max_pop
    return map2vec / map2vec.max() if map2vec.max() > 0.0 else map2vec


def construct_map2vec_full_scale(a_list, polygon_size):
    """
    This function is similar to the above BUT it builds map2vec WITHOUT removing redundant polygons.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: size of each polygon in degrees i.e 1x1 or 2x2
    :return: map2vec (full scale) i.e. without removing redundant polygons
    """
    map2vec = np.zeros(int(360 / polygon_size) * int(180 / polygon_size))
    if len(a_list) == 0:
        return map2vec
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        map2vec[index] += float(max(s[2], 1)) / max_pop
    return map2vec / map2vec.max() if map2vec.max() > 0.0 else map2vec


def merge_lists(lists):
    """
    Utility function to merge multiple lists.
    :param lists: a list of lists to be merged
    :return: one single list with all items from above list of lists
    """
    out = []
    for l in lists:
        out.extend(l)
    return out


def populate_sql():
    """
    Create and populate the sqlite3 database with GeoNames data. Requires Geonames dump.
    No need to run this function, I share the database as a separate dump on GitHub (see link).
    """
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
    """
    Utility function to eliminate code duplication. Nothing of much interest, methinks.
    :param class_code: Geonames code for the class of location
    :param feat_code: Geonames code for the feature type of an database entry
    :param p_map: dictionary mapping feature codes to estimated population
    :param pop: population count
    :return: population (modified if class code is one of A, P or L.
    """
    # interested = ["ADM1", "ADM2", "ADM3", "PCL", "LK", "SEA", "AREA", "CONT",
    #               "CST", "TRB", "ISL", "DSRT", "PLN", "PEN", "MT", "FRST"]
    if pop == 0 and class_code in ["A", "P", "L"]:
        pop = p_map.get(feat_code, 0)
    return pop


def generate_training_data():
    """
    Prepare Wikipedia training data. Please download the required files from GitHub.
    Files: geonames.db and geowiki.txt both inside the data folder (see README)
    Alternatively, create your own with http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
    """
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load(u'en')
    padding = nlp(u"0")[0]
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
                            near_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, d.i - CONTEXT_LENGTH / 2):d.i]], True, padding) \
                                       + pad_list(CONTEXT_LENGTH / 2, [x for x in doc[d.i + len(target): d.i + len(target) + CONTEXT_LENGTH / 2]], False, padding)
                            far_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, d.i - CONTEXT_LENGTH):max(0, d.i - CONTEXT_LENGTH / 2)]], True, padding) \
                                      + pad_list(CONTEXT_LENGTH / 2, [x for x in doc[d.i + len(target) + CONTEXT_LENGTH / 2: d.i + len(target) + CONTEXT_LENGTH]], False, padding)
                            near_out, far_out = [], []
                            location = u""
                            for (out_list, in_list, is_near) in [(near_out, near_inp, True), (far_out, far_inp, False)]:
                                for index, item in enumerate(in_list):
                                    if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
                                        if item.ent_iob_ == u"B" and item.text.lower() == u"the":
                                            out_list.append(item.text.lower())
                                        else:
                                            location += item.text + u" "
                                            out_list.append(u"**LOC**" + item.text.lower())
                                    elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY",
                                                            u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
                                        out_list.append(u'0')
                                    elif item.is_punct:
                                        out_list.append(u'0')
                                    elif item.is_digit or item.like_num:
                                        out_list.append(u'0')
                                    elif item.like_email:
                                        out_list.append(u'0')
                                    elif item.like_url:
                                        out_list.append(u'0')
                                    elif item.is_stop:
                                        out_list.append(u'0')
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
                                                if in_list[i + offset].is_alpha and location != u" ".join(target) else u'0'
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
    """
    Create evaluation data from text files. See README for formatting and download instructions.
    :param corpus: name of the dataset such as LGL, GEOVIRUS or WIKTOR
    :param file_name: an affix, in case you're creating several versions of the same dataset
    """
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load(u'en')
    padding = nlp(u"0")[0]
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
                        near_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, d.i - CONTEXT_LENGTH / 2):d.i]], True, padding) \
                                 + pad_list(CONTEXT_LENGTH / 2, [x for x in doc[d.i + len(target): d.i + len(target) + CONTEXT_LENGTH / 2]], False, padding)
                        far_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, d.i - CONTEXT_LENGTH):max(0, d.i - CONTEXT_LENGTH / 2)]], True, padding) \
                                + pad_list(CONTEXT_LENGTH / 2, [x for x in doc[d.i + len(target) + CONTEXT_LENGTH / 2: d.i + len(target) + CONTEXT_LENGTH]], False, padding)
                        near_out, far_out = [], []
                        location = u""
                        for (out_list, in_list, is_near) in [(near_out, near_inp, True), (far_out, far_inp, False)]:
                            for index, item in enumerate(in_list):
                                if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
                                    if item.ent_iob_ == u"B" and item.text.lower() == u"the":
                                        out_list.append(item.text.lower())
                                    else:
                                        location += item.text + u" "
                                        out_list.append(u"**LOC**" + item.text.lower())
                                elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY",
                                                        u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
                                    out_list.append(u'0')
                                elif item.is_punct:
                                    out_list.append(u'0')
                                elif item.is_digit or item.like_num:
                                    out_list.append(u'0')
                                elif item.like_email:
                                    out_list.append(u'0')
                                elif item.like_url:
                                    out_list.append(u'0')
                                elif item.is_stop:
                                    out_list.append(u'0')
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
                                                if in_list[i + offset].is_alpha and location != u" ".join(target) else u'0'
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
    """
    Display 2D array data with a title. Optional: log for better visualisation of small values.
    :param x: 2D numpy array you want to visualise
    :param title: of the chart because it's nice to have one :-)
    :param log: True in order to log the values and make for better visualisation, False for raw numbers
    """
    if log:
        x = np.log10(x)
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap', ['lightgrey', 'darkgrey', 'dimgrey', 'black'])
    cmap.set_bad(color='white')
    img = pyplot.imshow(x, cmap=cmap, interpolation='nearest')
    pyplot.colorbar(img, cmap=cmap)
    plt.title(title)
    # plt.savefig(title + u".png", dpi=200, transparent=True)  # Uncomment to save to file
    plt.show()


def generate_vocabulary(path, min_words, min_entities):
    """
    Prepare the vocabulary for training/testing. This function is to be called on generated data only, not plain text.
    :param path: to the file from which to build
    :param min_words: occurrence for inclusion in the vocabulary
    :param min_entities: occurrence for inclusion in the vocabulary
    """
    vocab_words, vocab_locations = {UNKNOWN, u'0'}, {UNKNOWN, u'0'}
    words, locations = [], []
    for f in [path]:  # You can also build the vocabulary from several files, just add to the list.
        training_file = codecs.open(f, u"r", encoding=u"utf-8")
        for line in training_file:
            line = line.strip().split("\t")
            words.extend([w for w in eval(line[2]) if u"**LOC**" not in w])  # NEAR WORDS
            words.extend([w for w in eval(line[3]) if u"**LOC**" not in w])  # FAR WORDS
            locations.extend([w for w in eval(line[2]) if u"**LOC**" in w])  # NEAR ENTITIES
            locations.extend([w for w in eval(line[3]) if u"**LOC**" in w])  # FAR ENTITIES

    words = Counter(words)
    for word in words:
        if words[word] > min_words:
            vocab_words.add(word)
    print(u"Words saved:", len(vocab_words))

    locations = Counter(locations)
    for location in locations:
        if locations[location] > min_entities:
            vocab_locations.add(location.replace(u"**LOC**", u""))
    print(u"Locations saved:", len(vocab_locations))

    vocabulary = vocab_words.union(vocab_locations)
    word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])
    cPickle.dump(word_to_index, open(u"data/w2i.pkl", "w"))


def generate_arrays_from_file(path, w2i, train=True):
    """
    Generator function for the FULL (SOTA) CNN + map2vec model in the paper. Uses all available data inputs.
    :param path: to the training file (see training data generation functions)
    :param w2i: the vocabulary set
    :param train: True is generating training data, false for test data
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        context_words, entities_strings, labels = [], [], []
        map2vec, target_string = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_map2vec([(float(line[0]), float(line[1]), 0)], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))

            near = [w if u"**LOC**" not in w else u'0' for w in eval(line[2])]
            far = [w if u"**LOC**" not in w else u'0' for w in eval(line[3])]
            context_words.append(far[:CONTEXT_LENGTH / 2] + near + far[CONTEXT_LENGTH / 2:])

            near = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else u'0' for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else u'0' for w in eval(line[3])]
            entities_strings.append(far[:CONTEXT_LENGTH / 2] + near + far[CONTEXT_LENGTH / 2:])

            # map2vec.append(construct_map2vec(sorted(eval(line[4]) + eval(line[6]) + eval(line[7]),
            #                key=lambda (a, b, c, d): c, reverse=True), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))
            # paper version above versus experimental setup below, map2vec is fully modular, remember? Try both!
            map2vec.append(construct_map2vec(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))
            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), True, u'0'))

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
                            np.asarray(entities_strings), np.asarray(map2vec), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                            np.asarray(entities_strings), np.asarray(map2vec), np.asarray(target_string)])

                context_words, entities_strings, labels = [], [], []
                map2vec, target_string = [], []

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
                        np.asarray(entities_strings), np.asarray(map2vec), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                        np.asarray(entities_strings), np.asarray(map2vec), np.asarray(target_string)])


def generate_arrays_from_file_lstm(path, w2i, train=True):
    """
    Generator for the context2vec model. Uses only lexical features.
    To replicate the map2vec + CONTEXT2VEC model from the paper, uncomment a few sections below
    and in the context2vec.py file. I hope it's clear enough :-) Email me if it isn't!
    :param path: to the training file (see training data generation functions)
    :param w2i: the vocabulary set
    :param train: True for training stage, False for testing stage
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        left, right, map2vec = [], [], []
        target_string, labels = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_map2vec([(float(line[0]), float(line[1]), 0)], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))

            near = [w.replace(u"**LOC**", u"") for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") for w in eval(line[3])]
            left.append(far[:CONTEXT_LENGTH / 2] + near[:CONTEXT_LENGTH / 2])
            right.append(near[CONTEXT_LENGTH / 2:] + far[CONTEXT_LENGTH / 2:])

            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), True, u'0'))

            # map2vec.append(construct_map2vec(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

            if counter % BATCH_SIZE == 0:
                for collection in [left, right, target_string]:
                    for x in collection:
                        for i, w in enumerate(x):
                            if w in w2i:
                                x[i] = w2i[w]
                            else:
                                x[i] = w2i[UNKNOWN]
                if train:
                    yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)], np.asarray(labels))
                    # yield ([np.asarray(left), np.asarray(right), np.asarray(map2vec), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)])
                    # yield ([np.asarray(left), np.asarray(right), np.asarray(map2vec), np.asarray(target_string)])

                left, right, map2vec = [], [], []
                target_string, labels = [], []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for collection in [left, right, target_string]:
                for x in collection:
                    for i, w in enumerate(x):
                        if w in w2i:
                            x[i] = w2i[w]
                        else:
                            x[i] = w2i[UNKNOWN]
            if train:
                yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)], np.asarray(labels))
                # yield ([np.asarray(left), np.asarray(right), np.asarray(map2vec), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)])
                # yield ([np.asarray(left), np.asarray(right), np.asarray(map2vec), np.asarray(target_string)])


def generate_strings_from_file(path):
    """
    Generator of labels, location names and context. Used for training and testing.
    :param path: to the training file (see training data generation functions)
    :return: Yields a list of tuples [(label, location name, context), ...]
    """
    while True:
        for line in codecs.open(path, "r", encoding="utf-8"):
            line = line.strip().split("\t")
            context = u" ".join(eval(line[2])) + u"*E*" + u" ".join(eval(line[5])) + u"*E*" + u" ".join(eval(line[3]))
            yield ((float(line[0]), float(line[1])), u" ".join(eval(line[5])).strip(), context)


def generate_arrays_from_file_map2vec(path, train=True, looping=True):
    """
    Generator for the plain map2vec model, works for MLP, Naive Bayes or Random Forest.
    :param path: to the training file (see training data generation functions)
    :param train: True for training phase, False for testing phase
    :param looping: True for continuous generation, False for one iteration.
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        labels, target_coord = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_map2vec([(float(line[0]), float(line[1]), 0, u'')], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))
            target_coord.append(construct_map2vec(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

            if counter % BATCH_SIZE == 0:
                if train:
                    yield ([np.asarray(target_coord)], np.asarray(labels))
                else:
                    yield ([np.asarray(target_coord)])

                labels = []
                target_coord = []

        if len(labels) > 0:
            # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            if train:
                yield ([np.asarray(target_coord)], np.asarray(labels))
            else:
                yield ([np.asarray(target_coord)])
        if not looping:
            break


def shrink_map2vec(polygon_size):
    """
    Remove polygons that only cover oceans. Dumps a dictionary of DB entries.
    :param polygon_size: the size of each polygon such as 1x1 or 3x3 degrees (integer)
    """
    map2vec = np.zeros((180 / polygon_size) * (360 / polygon_size),)
    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        lat, lon = float(line[4]), float(line[5])
        index = coord_to_index((lat, lon), polygon_size=polygon_size)
        map2vec[index] += 1.0
    cPickle.dump(map2vec, open(u"map2vec.pkl", "w"))


def oracle(path):
    """
    Calculate the Oracle (best possible given your database) performance for a given dataset.
    Prints the Oracle scores including mean, media, AUC and acc@161.
    :param path: file path to evaluate
    """
    final_errors = []
    conn = sqlite3.connect(u'../data/geonames.db')
    for line in codecs.open(path, "r", encoding="utf-8"):
        line = line.strip().split("\t")
        coordinates = (float(line[0]), float(line[1]))
        best_candidate = []
        for candidate in get_coordinates(conn.cursor(), u" ".join(eval(line[5])).strip()):
            best_candidate.append(great_circle(coordinates, (float(candidate[0]), float(candidate[1]))).km)
        final_errors.append(sorted(best_candidate)[0])
    print_stats(final_errors)


def prepare_geocorpora():
    f = codecs.open("data/geocorpora.txt", "w", "utf-8")
    for line in codecs.open("data/GeoCorpora_ann.tsv", encoding="utf-8"):
        if line.strip() == "":
            f.write("\n")
            continue
        if line.startswith(u"130"):
            print ""
        line = line.strip().split("\t")
        f.write(line[5] + ",," + line[3] + ",," + line[7] + ",," + line[8] + ",," + line[2] + ",," + str(int(line[2]) + len(line[3])) + "||\n")
    f.close()
    counter = 0
    for line in codecs.open("data/GeoCorpora.tsv", encoding="utf-8"):
        f = codecs.open("../data/geocorpora/" + str(counter), "w", "utf-8")
        line = line.split("\t")
        f.write(line[15].strip())
        f.close()
        counter += 1

# --------------------------------------------- INVOKE FUNCTIONS ---------------------------------------------------
# prepare_geocorpora()
# print get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"dublin")
# generate_training_data()
# generate_evaluation_data(corpus="geovirus", file_name="")
# generate_vocabulary(path=u"../data/train_wiki.txt", min_words=9, min_entities=1)
# shrink_map2vec(2)
# oracle(u"data/eval_geovirus_gold.txt")
# conn = sqlite3.connect('../data/geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"darfur", u"[(13.5, 23.5, 0), (44.05135, -94.83804, 106)]"))
# c.execute("DELETE FROM GEO WHERE name = 'darfur'")
# conn.commit()
# print index_to_coord(8177, 2)
# populate_sql()

# -------- CREATE MAPS (mapping from 64,000/16,200 polygons to 23,002, 7,821) ------------
# l2v = list(cPickle.load(open(u"data/geonames_1x1.pkl")))
# zeros = dict([(i, v) for i, v in enumerate(l2v) if v > 0])  # isolate the non zero values
# zeros = dict([(i, v) for i, v in enumerate(zeros)])         # replace counts with indices
# zeros = dict([(v, i) for (i, v) in zeros.iteritems()])      # reverse keys and values
# cPickle.dump(zeros, open(u"data/1x1_encode_map.pkl", "w"))

# ------- VISUALISE THE WHOLE DATABASE ----------
# l2v = np.reshape(l2v, newshape=((180 / 1), (360 / 1)))
# visualise_2D_grid(l2v, "Geonames Database", True)

# -------- CREATE OUTLIERS (polygons outside of map2vec) MAP --------
# filtered = [i for i, v in enumerate(l2v) if v > 0]
# the_rest = [i for i, v in enumerate(l2v) if v == 0]
# poly_size = 2
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
# cPickle.dump(dict_rest, open(u"data/2x2_outliers_map.pkl", "w"))

# ------ PROFILING SETUP -----------
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

# ----------- VISUALISATION OF DIFFERENT LOCATIONS -------------
# print len(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Melbourne"))
# coord = get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Ebola")
# print coord
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Victoria"))
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Newcastle"))
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Perth"))
# coord = sorted(coord, key=lambda (a, b, c, d): c, reverse=True)
# x = construct_map2vec_full_scale(coord, polygon_size=3)
# x = np.reshape(x, newshape=((180 / 3), (360 / 3)))
# visualise_2D_grid(x, "Melbourne", True)

# ---------- DUMP DATABASE ------
# import sqlite3
#
# con = sqlite3.connect('../data/geonames.db')
# with codecs.open('dump.sql', 'w', 'utf-8') as f:
#     for line in con.iterdump():
#         f.write('%s\n' % line)
# -------------------------------


# gold = {}
# out = codecs.open("data/geovirus.txt", mode="w", encoding="utf-8")
# common = set([2049, 2054, 2057, 2058, 2060, 13, 2065, 2067, 2068, 22, 27, 33, 34, 35, 2089, 42, 44, 2093, 46, 2096, 2097, 51, 52, 53, 54, 55, 56, 2106, 59, 2108, 2109, 67, 2117, 2119, 75, 2126, 79, 80, 2129, 2130, 83, 85, 86, 2136, 2137, 90, 91, 92, 2142, 99, 2148, 2150, 2151, 106, 107, 108, 110, 111, 113, 114, 2163, 116, 117, 2166, 119, 123, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 143, 146, 147, 153, 154, 155, 160, 161, 162, 165, 166, 171, 176, 179, 180, 182, 183, 185, 186, 187, 188, 189, 191, 192, 195, 198, 2081, 201, 202, 207, 208, 211, 213, 214, 215, 222, 224, 225, 227, 230, 231, 232, 236, 242, 243, 244, 245, 246, 252, 2090, 254, 257, 262, 263, 264, 265, 268, 270, 272, 273, 276, 277, 278, 282, 283, 284, 285, 287, 288, 289, 291, 293, 294, 295, 296, 300, 301, 302, 304, 305, 311, 315, 2101, 328, 329, 330, 332, 333, 334, 337, 342, 344, 345, 346, 347, 351, 352, 353, 354, 357, 359, 361, 362, 363, 364, 365, 368, 369, 371, 373, 375, 379, 380, 383, 385, 386, 387, 388, 391, 392, 393, 394, 395, 396, 397, 399, 400, 401, 2115, 405, 2133, 2155, 2158, 2159, 2162, 2164, 2165, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1182, 1183, 1184, 1186, 1188, 1189, 1190, 1191, 1196, 1197, 1203, 1209, 1211, 1213, 1215, 1217, 1219, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1237, 1238, 1239, 1241, 1242, 1243, 1246, 1247, 1249, 1256, 1257, 1259, 1260, 1262, 1264, 1267, 1270, 1272, 1273, 1278, 1279, 1280, 1281, 1282, 1283, 1298, 1303, 1304, 1307, 1308, 1309, 1333, 1335, 1338, 1347, 1348, 1354, 1355, 1358, 1360, 1366, 1367, 1374, 1384, 1386, 1387, 1396, 1397, 1398, 1401, 1402, 1404, 1408, 1409, 1410, 1411, 1418, 1419, 1420, 1421, 1422, 1423, 1432, 1433, 1435, 1436, 1437, 1441, 1443, 1449, 2094, 1455, 1456, 1457, 1458, 1461, 1462, 1464, 1466, 1467, 1468, 1469, 1471, 1473, 1474, 1480, 1481, 1482, 1483, 1486, 1487, 1490, 1502, 1506, 1511, 1514, 1515, 1519, 1520, 1522, 1523, 1524, 1526, 1527, 1531, 1533, 1534, 1538, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1554, 1556, 1557, 1558, 1559, 1565, 1566, 1568, 1569, 1576, 1581, 1582, 1583, 1584, 1586, 1587, 1588, 1589, 1591, 1592, 1593, 1594, 1595, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1608, 1610, 1611, 1612, 1615, 1616, 1623, 1627, 1630, 1632, 1633, 1635, 1636, 1637, 1642, 1643, 1644, 1650, 1652, 1660, 1661, 1662, 1663, 1665, 1667, 1668, 1670, 1671, 1673, 1678, 1683, 1685, 1686, 1690, 1691, 1693, 1697, 1698, 1699, 1700, 1701, 1702, 1705, 1707, 1711, 1716, 1717, 1722, 1723, 1725, 1731, 1734, 1735, 1737, 1738, 1739, 1741, 1742, 1743, 1744, 1745, 1748, 1753, 1755, 1759, 1761, 1766, 1767, 1770, 1773, 1775, 1776, 1785, 1788, 1789, 1791, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1801, 1802, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1813, 1815, 1816, 1817, 1819, 1820, 1822, 1823, 1825, 1826, 1829, 1832, 1833, 1836, 1838, 1839, 1842, 1844, 1849, 1854, 1855, 1856, 1873, 1882, 1883, 1884, 1885, 1902, 1903, 1904, 1905, 1906, 1907, 1909, 1919, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1934, 1937, 1938, 1939, 1941, 1943, 1945, 1947, 1949, 1954, 1956, 1958, 1960, 1962, 1963, 1965, 1966, 1967, 1974, 1978, 1979, 1980, 1981, 1982, 1985, 1990, 2000, 2002, 2005, 2006, 2008, 2009, 2011, 2012, 2014, 2022, 2027, 2029, 2030, 2031, 2032, 2037, 2040, 2042, 2045, 2047])
# index = 0
# for line in codecs.open("data/geovirus_gold.txt", encoding="utf-8"):
#     for l in line.split("||")[:-1]:
#         if index in common:
#             out.write(l + u"||")
#         index += 1
#     out.write(u"\n")

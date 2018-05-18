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
CONTEXT_LENGTH = 200  # each side of target entity
UNKNOWN = u"<unknown>"
EMBEDDING_DIMENSION = 50
TARGET_LENGTH = 15
ENCODING_MAP_1x1 = cPickle.load(open(u"data/1x1_encode_map.pkl"))      # We need these maps
ENCODING_MAP_2x2 = cPickle.load(open(u"data/2x2_encode_map.pkl"))      # and the reverse ones
REVERSE_MAP_1x1 = cPickle.load(open(u"data/1x1_reverse_map.pkl"))      # to handle the used and
REVERSE_MAP_2x2 = cPickle.load(open(u"data/2x2_reverse_map.pkl"))      # unused map_vector polygons.
OUTLIERS_MAP_1x1 = cPickle.load(open(u"data/1x1_outliers_map.pkl"))    # Outliers are redundant polygons that
OUTLIERS_MAP_2x2 = cPickle.load(open(u"data/2x2_outliers_map.pkl"))    # have been removed but must also be handled.
# -------- GLOBAL CONSTANTS AND VARIABLES -------- #


def print_stats(accuracy):
    """
    Prints Mean, Median, AUC and acc@161km for the list.
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
    Utility function that pads a list with any given padding.
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
    Convert coordinates into an array (world representation) index. Use that to modify map_vector polygon value.
    :param coordinates: (latitude, longitude) to convert to the map vector index
    :param polygon_size: integer size of the polygon? i.e. the resolution of the world
    :return: index pointing into map_vector array
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
    Convert index (output of the prediction model) back to coordinates.
    :param index: of the polygon/tile in map_vector array (given by model prediction)
    :param polygon_size: size of each polygon/tile i.e. resolution of the world
    :return: pair of (latitude, longitude)
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


def construct_map_vector(a_list, polygon_size, mapping, outliers):
    """
    Build the map_vector vector representation from a_list of location data.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: what's the resolution? size of each polygon in degrees.
    :param mapping: one of the transformation maps 1x1 or 2x2
    :param outliers: the outlier map, 1x1 or 2x2 (must match resolution or mapping above)
    :return: map_vector representation
    """
    map_vector = np.zeros(len(mapping), )
    if len(a_list) == 0:
        return map_vector
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        if index in mapping:
            index = mapping[index]
        else:
            index = mapping[outliers[index]]
        map_vector[index] += float(max(s[2], 1)) / max_pop
    return map_vector / map_vector.max() if map_vector.max() > 0.0 else map_vector


def construct_map_vector_full_scale(a_list, polygon_size):
    """
    This function is similar to the above BUT it builds map_vector WITHOUT removing redundant polygons.
    :param a_list: of tuples [(latitude, longitude, population, feature_code), ...]
    :param polygon_size: size of each polygon in degrees i.e 1x1 or 2x2
    :return: map_vector (full scale) i.e. without removing redundant polygons, used for visualisation in 2D
    """
    map_vector = np.zeros(int(360 / polygon_size) * int(180 / polygon_size))
    if len(a_list) == 0:
        return map_vector
    max_pop = a_list[0][2] if a_list[0][2] > 0 else 1
    for s in a_list:
        index = coord_to_index((s[0], s[1]), polygon_size)
        map_vector[index] += float(max(s[2], 1)) / max_pop
    return map_vector / map_vector.max() if map_vector.max() > 0.0 else map_vector


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
    nlp = spacy.load(u'en')  # or spacy.load(u'en_core_web_lg') depending on your Spacy Download (simple, full)
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
    cPickle.dump(word_to_index, open(u"data/words2index.pkl", "w"))


def generate_arrays_from_file(path, words_to_index, train=True):
    """
    Generator function for the FULL (SOTA) CNN + map_vector model in the paper. Uses all available data inputs.
    :param path: to the training file (see training data generation functions)
    :param words_to_index: the vocabulary set
    :param train: True is generating training data, false for test data
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        context_words, entities_strings, labels = [], [], []
        map_vector, target_string = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_map_vector([(float(line[0]), float(line[1]), 0)], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))

            near = [w if u"**LOC**" not in w else u'0' for w in eval(line[2])]
            far = [w if u"**LOC**" not in w else u'0' for w in eval(line[3])]
            context_words.append(far[:CONTEXT_LENGTH / 2] + near + far[CONTEXT_LENGTH / 2:])

            near = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else u'0' for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") if u"**LOC**" in w else u'0' for w in eval(line[3])]
            entities_strings.append(far[:CONTEXT_LENGTH / 2] + near + far[CONTEXT_LENGTH / 2:])

            # map_vector.append(construct_map_vector(sorted(eval(line[4]) + eval(line[6]) + eval(line[7]),
            #                key=lambda (a, b, c, d): c, reverse=True), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))
            # paper version above versus experimental setup below, map_vector is fully modular, remember? Try both!
            map_vector.append(construct_map_vector(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))
            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), True, u'0'))

            if counter % BATCH_SIZE == 0:
                for collection in [context_words, entities_strings, target_string]:
                    for x in collection:
                        for i, w in enumerate(x):
                            if w in words_to_index:
                                x[i] = words_to_index[w]
                            else:
                                x[i] = words_to_index[UNKNOWN]
                if train:
                    yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                            np.asarray(entities_strings), np.asarray(map_vector), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                            np.asarray(entities_strings), np.asarray(map_vector), np.asarray(target_string)])

                context_words, entities_strings, labels = [], [], []
                map_vector, target_string = [], []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for collection in [context_words, entities_strings, target_string]:
                for x in collection:
                    for i, w in enumerate(x):
                        if w in words_to_index:
                            x[i] = words_to_index[w]
                        else:
                            x[i] = words_to_index[UNKNOWN]
            if train:
                yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                        np.asarray(entities_strings), np.asarray(map_vector), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
                        np.asarray(entities_strings), np.asarray(map_vector), np.asarray(target_string)])


def generate_arrays_from_file_lstm(path, words_to_index, train=True):
    """
    Generator for the context2vec model. Uses only lexical features.
    To replicate the map_vector + CONTEXT2VEC model from the paper, uncomment a few sections below
    and in the context2vec.py file. I hope it's clear enough :-) Email me if it isn't!
    :param path: to the training file (see training data generation functions)
    :param words_to_index: the vocabulary set
    :param train: True for training stage, False for testing stage
    """
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        left, right, map_vector = [], [], []
        target_string, labels = [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            labels.append(construct_map_vector([(float(line[0]), float(line[1]), 0)], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))

            near = [w.replace(u"**LOC**", u"") for w in eval(line[2])]
            far = [w.replace(u"**LOC**", u"") for w in eval(line[3])]
            left.append(far[:CONTEXT_LENGTH / 2] + near[:CONTEXT_LENGTH / 2])
            right.append(near[CONTEXT_LENGTH / 2:] + far[CONTEXT_LENGTH / 2:])

            target_string.append(pad_list(TARGET_LENGTH, eval(line[5]), True, u'0'))

            # map_vector.append(construct_map_vector(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

            if counter % BATCH_SIZE == 0:
                for collection in [left, right, target_string]:
                    for x in collection:
                        for i, w in enumerate(x):
                            if w in words_to_index:
                                x[i] = words_to_index[w]
                            else:
                                x[i] = words_to_index[UNKNOWN]
                if train:
                    yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)], np.asarray(labels))
                    # yield ([np.asarray(left), np.asarray(right), np.asarray(map_vector), np.asarray(target_string)], np.asarray(labels))
                else:
                    yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)])
                    # yield ([np.asarray(left), np.asarray(right), np.asarray(map_vector), np.asarray(target_string)])

                left, right, map_vector = [], [], []
                target_string, labels = [], []

        if len(labels) > 0:  # This block is only ever entered at the end to yield the final few samples. (< BATCH_SIZE)
            for collection in [left, right, target_string]:
                for x in collection:
                    for i, w in enumerate(x):
                        if w in words_to_index:
                            x[i] = words_to_index[w]
                        else:
                            x[i] = words_to_index[UNKNOWN]
            if train:
                yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)], np.asarray(labels))
                # yield ([np.asarray(left), np.asarray(right), np.asarray(map_vector), np.asarray(target_string)], np.asarray(labels))
            else:
                yield ([np.asarray(left), np.asarray(right), np.asarray(target_string)])
                # yield ([np.asarray(left), np.asarray(right), np.asarray(map_vector), np.asarray(target_string)])


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


def generate_arrays_from_file_map_vector(path, train=True, looping=True):
    """
    Generator for the plain map_vector model, works for MLP, Naive Bayes or Random Forest. Table 2 in the paper.
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
            labels.append(construct_map_vector([(float(line[0]), float(line[1]), 0, u'')], 2, ENCODING_MAP_2x2, OUTLIERS_MAP_2x2))
            target_coord.append(construct_map_vector(eval(line[4]) + eval(line[6]) + eval(line[7]), 1, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1))

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


def shrink_map_vector(polygon_size):
    """
    Remove polygons that only cover oceans. Dumps a dictionary of DB entries.
    :param polygon_size: the size of each polygon such as 1x1 or 2x2 or 3x3 degrees (integer)
    """
    map_vector = np.zeros((180 / polygon_size) * (360 / polygon_size),)
    for line in codecs.open(u"../data/allCountries.txt", u"r", encoding=u"utf-8"):
        line = line.split("\t")
        lat, lon = float(line[4]), float(line[5])
        index = coord_to_index((lat, lon), polygon_size=polygon_size)
        map_vector[index] += 1.0
    cPickle.dump(map_vector, open(u"mapvec_shrink.pkl", "w"))


def oracle(path):
    """
    Calculate the Oracle (best possible given your database) performance for a given dataset.
    Prints the Oracle scores including mean, median, AUC and acc@161.
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


# --------------------------------------------- INVOKE FUNCTIONS ---------------------------------------------------
# prepare_geocorpora()
# print get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"dublin")
# generate_training_data()
# generate_evaluation_data(corpus="geovirus", file_name="")
# generate_vocabulary(path=u"../data/train_wiki.txt", min_words=9, min_entities=1)
# shrink_map_vector(2)
# oracle(u"data/eval_geovirus_gold.txt")
# conn = sqlite3.connect('../data/geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"darfur", u"[(13.5, 23.5, 0), (44.05135, -94.83804, 106)]"))
# c.execute("DELETE FROM GEO WHERE name = 'darfur'")
# conn.commit()
# print index_to_coord(8177, 2)
# populate_sql()

# -------- CREATE MAPS (mapping from 64,000/16,200 polygons to 23,002, 7,821) ------------
# map_vector = list(cPickle.load(open(u"data/1x1_geonames.pkl")))
# zeros = dict([(i, v) for i, v in enumerate(map_vector) if v > 0])  # isolate the non zero values
# zeros = dict([(i, v) for i, v in enumerate(zeros)])                # replace counts with indices
# zeros = dict([(v, i) for (i, v) in zeros.iteritems()])             # reverse keys and values
# cPickle.dump(zeros, open(u"data/1x1_encode_map.pkl", "w"))

# ------- VISUALISE THE WHOLE DATABASE ----------
# map_vector = np.reshape(map_vector, newshape=((180 / 1), (360 / 1)))
# visualise_2D_grid(map_vector, "Geonames Database", True)

# -------- CREATE OUTLIERS (polygons outside of map_vector) MAP --------
# filtered = [i for i, v in enumerate(map_vector) if v > 0]
# the_rest = [i for i, v in enumerate(map_vector) if v == 0]
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
# coord = get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Giza")
# print coord
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Giza Plateau"))
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Cairo"))
# coord.extend(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Egypt"))
# coord = sorted(coord, key=lambda (a, b, c, d): c, reverse=True)
# x = construct_map_vector_full_scale(coord, polygon_size=2)
# x = np.reshape(x, newshape=((180 / 2), (360 / 2)))
# visualise_2D_grid(x, "Giza, Giza Plateau, Egypt, Cairo", True)

# ---------- DUMP DATABASE ------
# import sqlite3
#
# con = sqlite3.connect('../data/geonames.db')
# with codecs.open('dump.sql', 'w', 'utf-8') as f:
#     for line in con.iterdump():
#         f.write('%s\n' % line)
# -------------------------------

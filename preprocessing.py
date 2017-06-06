# -*- coding: utf-8 -*-
import codecs
import cPickle
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import numpy as np
import sqlite3
from matplotlib import pyplot, colors
from scipy.spatial.distance import euclidean

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
    # print u"AUC NO_LOG = ", np.trapz(np.exp(accuracy)) / (20039 * (len(accuracy) - 1))  # Trapezoidal rule.
    print("==============================================================================================")


def pad_list(size, a_list, from_left):
    """"""
    while len(a_list) < size:
        if from_left:
            a_list = [0.0] + a_list
        else:
            a_list += [0.0]
    return a_list


def coord_to_index(coordinates):
    """"""
    latitude = float(coordinates[0]) - 90 if float(coordinates[0]) != -90 else -179.99
    longitude = float(coordinates[1]) + 180 if float(coordinates[1]) != 180 else 359.99
    if longitude < 0:
        longitude = -longitude
    if latitude < 0:
        latitude = -latitude
    x = (360 / GRID_SIZE) * (int(latitude) / GRID_SIZE)
    y = int(longitude) / GRID_SIZE
    return x + y if 0 <= x + y <= (360 / GRID_SIZE) * (180 / GRID_SIZE) else Exception(u"Shock horror!!")


def index_to_coord(index):
    """"""
    x = int(index / (360 / GRID_SIZE))
    y = index % (360 / GRID_SIZE)
    if x > (90 / GRID_SIZE):
        x = -(x - (90 / GRID_SIZE)) * GRID_SIZE  # - GRID_SIZE / 2.0
    else:
        x = ((90 / GRID_SIZE) - x) * GRID_SIZE  # + GRID_SIZE / 2.0
    if y < (180 / GRID_SIZE):
        y = -((180 / GRID_SIZE) - y) * GRID_SIZE  # - GRID_SIZE / 2.0
    else:
        y = (y - (180 / GRID_SIZE)) * GRID_SIZE  # + GRID_SIZE / 2.0
    return x, y


def get_coordinates(con, loc_name, pop_only):
    """"""
    result = con.execute(u"SELECT METADATA FROM GEO WHERE NAME = ?", (loc_name.lower(), )).fetchone()
    if result:
        result = eval(result[0])
        if pop_only:
            if max(result, key=lambda(a, b, c, d): c)[2] == 0:
                return result
            else:
                return [r for r in result if r[2] > 0]
                # return [r for r in result if r[3] in [u'A', u'P']]
        else:
            return result
    else:
        return []


def construct_1D_grid(a_list, use_pop, is_y, smoothing):
    """"""
    g = np.zeros((360 / GRID_SIZE) * (180 / GRID_SIZE))
    for s in a_list:
        index = coord_to_index((s[0], s[1]))
        if use_pop:
            g[index] += 1 + s[2]
            # visualise_2D_grid(np.reshape(g, (180 / GRID_SIZE, 360 / GRID_SIZE)), "Before")
            # apply_smoothing(g, index, 1 + s[2], smoothing)
            # visualise_2D_grid(np.reshape(g, (180 / GRID_SIZE, 360 / GRID_SIZE)), "After")
        else:
            g[index] += 1
            # visualise_2D_grid(np.reshape(g, (180 / GRID_SIZE, 360 / GRID_SIZE)), "Before")
            # apply_smoothing(g, index, 1, smoothing)
            # visualise_2D_grid(np.reshape(g, (180 / GRID_SIZE, 360 / GRID_SIZE)), "After")
    if is_y:
        return g / sum(g)  # FOR Y LABELS
    else:
        return g / max(g) if max(g) > 0.0 else g  # FOR REST OF THE GRID


def apply_smoothing(g, index, value, smoothing):
    """"""
    grid_width = 360 / GRID_SIZE
    if index % grid_width > 0:  # LEFT
        g[index - 1] += value * smoothing
    if index % grid_width != grid_width - 1:  # RIGHT
        g[index + 1] += value * smoothing
    if index >= grid_width:  # UP
        g[index - grid_width] += value * smoothing
    if index < len(g) - grid_width:  # DOWN
        g[index + grid_width] += value * smoothing
    if index >= grid_width and index % grid_width > 0:  # NORTH WEST
        g[index - 1 - grid_width] += value * smoothing
    if index >= grid_width and index % grid_width != grid_width - 1:  # NORTH EAST
        g[index + 1 - grid_width] += value * smoothing
    if index < len(g) - grid_width and index % grid_width > 0:  # SOUTH WEST
        g[index - 1 + grid_width] += value * smoothing
    if index < len(g) - grid_width and index % grid_width != grid_width - 1:  # SOUTH EAST
        g[index + 1 + grid_width] += value * smoothing


def construct_2D_grid(a_list, use_pop):
    """"""
    g = np.zeros(((180 / GRID_SIZE), (360 / GRID_SIZE)))
    for s in a_list:
        index = coord_to_index((s[0], s[1]))
        x = int(int(index / (360 / GRID_SIZE)) / GRID_SIZE)
        y = int(int(index % (360 / GRID_SIZE)) / GRID_SIZE)
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
    f = codecs.open(u"../data/allCountries.txt", "r", encoding="utf-8")

    for line in f:
        line = line.split("\t")
        for name in [line[1], line[2]] + line[3].split(","):
            name = name.lower()
            if len(name) != 0:
                if name in geo_names:
                    geo_names[name].add((float(line[4]), float(line[5]), int(line[14]), line[6]))
                else:
                    geo_names[name] = {(float(line[4]), float(line[5]), int(line[14]), line[6])}

    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    # c.execute("CREATE TABLE GEO (NAME VARCHAR(100) PRIMARY KEY NOT NULL, METADATA VARCHAR(5000) NOT NULL);")
    c.execute(u"DELETE FROM GEO")
    conn.commit()

    for gn in geo_names:
        c.execute(u"INSERT INTO GEO VALUES (?, ?)", (gn, str(list(geo_names[gn]))))
    print(u"Entries saved:", len(geo_names))
    conn.commit()
    conn.close()


def generate_training_data(context):
    """Prepare Wikipedia training data."""
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load('en')
    f = codecs.open(u"../data/geowiki.txt", "r", encoding="utf-8")
    o = codecs.open(u"../data/train_wiki.txt", "w", encoding="utf-8")
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
                            left = doc[max(0, d.i - context):d.i]
                            right = doc[d.i + len(target): d.i + len(target) + context]
                            l, r = [], []
                            location = u""
                            for (out_list, in_list) in [(l, left), (r, right)]:
                                for index, item in enumerate(in_list):
                                    if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
                                        if item.ent_iob_ == "B" and item.text.lower() == u"the":
                                            out_list.append(u"0.0")
                                        else:
                                            location += item.text + u" "
                                            out_list.append(u"0.0")
                                    elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY"
                                                            u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
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
                                        out_list.append(item.lemma_)
                                    if location.strip() != u"" and (item.ent_type == 0 or index == len(in_list) - 1):
                                        if location.strip() != u" ".join(target):
                                            coords = get_coordinates(c, location.strip(), pop_only=True)
                                            if len(coords) > 0:
                                                locations.append(coords)
                                            else:
                                                offset = 1 if index == len(in_list) - 1 else 0
                                                for i in range(index - len(location.split()), index):
                                                    out_list[i + offset] = in_list[i + offset].lemma_ if not in_list[i + offset].is_punct else u"0.0"
                                        location = u""
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
                            if limit > 24:
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


def generate_evaluation_data(corpus, file_name, context):
    """Prepare WikToR and LGL data. Only the subsets i.e. (2202 WIKTOR, 787 LGL)"""
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    nlp = spacy.load('en')
    directory = u"../data/" + corpus + "/"
    o = codecs.open(u"data/eval_" + corpus + file_name + ".txt", "w", encoding="utf-8")
    line_no = 0 if corpus == "lgl" else -1

    for line in codecs.open("data/" + corpus + file_name + ".txt", "r", encoding="utf-8"):
        line_no += 1
        if len(line.strip()) == 0:
            continue
        for toponym in line.split("||")[:-1]:
            captured = False
            doc = nlp(codecs.open(directory + str(line_no), "r", encoding="utf-8").read())
            toponym = toponym.split(",,")
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
                        left = doc[max(0, d.i - context):d.i]
                        right = doc[d.i + len(target): d.i + len(target) + context]
                        l, r = [], []
                        location = u""
                        for (out_list, in_list) in [(l, left), (r, right)]:
                            for index, item in enumerate(in_list):
                                if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
                                    if item.ent_iob_ == "B" and item.text.lower() == u"the":
                                        out_list.append(u"0.0")
                                    else:
                                        location += item.text + u" "
                                        out_list.append(u"0.0")
                                elif item.ent_type_ in [u"PERSON", u"DATE", u"TIME", u"PERCENT", u"MONEY"
                                                        u"QUANTITY", u"CARDINAL", u"ORDINAL"]:
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
                                    out_list.append(item.lemma_)
                                if location.strip() != u"" and (item.ent_type == 0 or index == len(in_list) - 1):
                                    if location.strip() != u" ".join(target):
                                        coords = get_coordinates(c, location.strip(), pop_only=True)
                                        if len(coords) > 0:
                                            locations.append(coords)
                                        else:
                                            offset = 1 if index == len(in_list) - 1 else 0
                                            for i in range(index - len(location.split()), index):
                                                out_list[i + offset] = in_list[i + offset].lemma_ if not in_list[i + offset].is_punct else u"0.0"
                                    location = u""
                        db_entry = toponym[0] if corpus == "lgl" else toponym[1]
                        target_grid = get_coordinates(c, db_entry, pop_only=True)
                        if len(target_grid) == 0:
                            raise Exception(u"No entry in the database!", db_entry)
                        entities_grid = merge_lists(locations)
                        o.write(lat + u"\t" + lon + u"\t" + str(l) + u"\t" + str(r) + u"\t")
                        o.write(str(target_grid) + u"\t" + str(entities_grid) + u"\t" + db_entry + u"\t" +
                        u" ".join([s.text for s in left]).strip() + u" ".join([s.text for s in right]).strip() + u"\n")
            if not captured:
                print line_no, line, target, start, end
    o.close()


def visualise_2D_grid(x, title):
    """"""
    x = x * 255
    cmap2 = colors.LinearSegmentedColormap.from_list('my_colormap', ['white', 'black', 'red'], 256)
    img2 = pyplot.imshow(np.log(x + 1), interpolation='nearest', cmap=cmap2)
    pyplot.colorbar(img2, cmap=cmap2)
    # plt.imshow(np.log(x + 1), cmap='gray', interpolation='nearest', vmin=0, vmax=np.log(255))
    plt.title(title)
    plt.show()


def generate_vocabulary():
    """Prepare the vocabulary for NN training."""
    vocabulary = {u"<unknown>", u"0.0"}
    temp = []
    for f in [u"../data/train_wiki.txt", u"data/eval_wiki_gold.txt", u"data/eval_lgl_gold.txt"]:
        training_file = codecs.open(f, "r", encoding="utf-8")
        for line in training_file:
            line = line.strip().split("\t")
            temp.extend(eval(line[2].lower()))
            temp.extend(eval(line[3].lower()))

    c = Counter(temp)
    for item in c:
        if c[item] > 5:
            vocabulary.add(item)
    cPickle.dump(vocabulary, open(u"data/vocabulary.pkl", "w"))
    print(u"Vocabulary Size:", len(vocabulary))


def generate_arrays_from_file(path, w2i, input_length, batch_size=64, train=True, oneDim=True):
    """"""
    while True:
        training_file = codecs.open(path, "r", encoding="utf-8")
        counter = 0
        X_L, X_R, X_E, X_T, Y = [], [], [], [], []
        for line in training_file:
            counter += 1
            line = line.strip().split("\t")
            if oneDim:
                Y.append(construct_1D_grid([(float(line[0]), float(line[1]), 0)], use_pop=False, is_y=True, smoothing=0.005))
            else:
                Y.append([(float(line[0]), float(line[1]))])
            X_L.append(pad_list(input_length, eval(line[2].lower()), from_left=True)[-input_length:])
            X_R.append(pad_list(input_length, eval(line[3].lower()), from_left=False)[:input_length])
            if oneDim:
                X_T.append(construct_1D_grid(eval(line[4]), use_pop=True, is_y=True, smoothing=0.005))
                X_E.append(construct_1D_grid(eval(line[5]), use_pop=False, is_y=False, smoothing=0.05))
            else:
                X_T.append(construct_2D_grid(eval(line[4]), use_pop=True))
                X_E.append(construct_2D_grid(eval(line[5]), use_pop=False))
            if counter % batch_size == 0:
                for x_l, x_r in zip(X_L, X_R):
                    for i, w in enumerate(x_l):
                        if w in w2i:
                            x_l[i] = w2i[w]
                        else:
                            x_l[i] = w2i[u"<unknown>"]
                    for i, w in enumerate(x_r):
                        if w in w2i:
                            x_r[i] = w2i[w]
                        else:
                            x_r[i] = w2i[u"<unknown>"]
                if train:
                    yield ([np.asarray(X_L), np.asarray(X_L), np.asarray(X_R), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)], np.asarray(Y))
                else:
                    yield ([np.asarray(X_L), np.asarray(X_L), np.asarray(X_R), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)])
                X_L, X_R, X_E, X_T, Y = [], [], [], [], []
        if len(Y) > 0:  # This block is only ever entered at the end to yield the final few samples. (< batch_size)
            for x_l, x_r in zip(X_L, X_R):
                for i, w in enumerate(x_l):
                    if w in w2i:
                        x_l[i] = w2i[w]
                    else:
                        x_l[i] = w2i[u"<unknown>"]
                for i, w in enumerate(x_r):
                    if w in w2i:
                        x_r[i] = w2i[w]
                    else:
                        x_r[i] = w2i[u"<unknown>"]
            if train:
                yield ([np.asarray(X_L), np.asarray(X_L), np.asarray(X_R), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)], np.asarray(Y))
            else:
                yield ([np.asarray(X_L), np.asarray(X_L), np.asarray(X_R), np.asarray(X_R), np.asarray(X_E), np.asarray(X_T)])


def generate_strings_from_file(path):
    """Returns Y, NAME and CONTEXT"""
    while True:
        for line in codecs.open(path, "r", encoding="utf-8"):
            line = line.strip().split("\t")
            yield ((float(line[0]), float(line[1])), line[6], line[7])


def compute_embedding_distances(W, dim):
    store = []
    W = np.reshape(W, (180 / GRID_SIZE, 360 / GRID_SIZE, dim))
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


def compute_pixel_similarity():
    distances_p = compute_embedding_distances(cPickle.load(open("data/W.pkl")), 801)

    store = []
    for r in range(180 / GRID_SIZE):
        store_c = []
        for c in range(360 / GRID_SIZE):
            store_c.append((r, c))
        store.append(store_c)

    distances_g = compute_embedding_distances(np.array(store), 2)

    correlations = []
    for p, g in zip(distances_p, distances_g):
        for cp, cg in zip(p, g):
            correlations.append(np.corrcoef(cp, cg))

    cPickle.dump(correlations, open(u"data/correlations.pkl", "w"))


# ----------------------------------------------INVOKE METHODS HERE----------------------------------------------------

# l = list(construct_1D_grid([(-81.8, -109.98333, 1000), (-80, -104.98333, 80), (-82.5, -102, 50)], use_pop=True, is_y=False))
# l = list(construct_1D_grid([(-61.8, -109.98333, 1000)], use_pop=True, is_y=True))
# print(l)
# l = np.reshape(l, (180 / GRID_SIZE, 360 / GRID_SIZE))
# visualise_2D_grid(l, "exp")
# print(list(construct_1D_grid([(90, -180, 0), (90, -170, 1000)], use_pop=True)))
# generate_training_data(context=150)
# generate_evaluation_data(corpus="wiki", file_name="_yahoo", context=200)
# index = coord_to_index((-6.43, -172.32), True)
# print(index, index_to_coord(index))
# generate_vocabulary()
# for word in generate_names_from_file("data/eval_lgl.txt"):
#     print word.strip()
# print(get_coordinates(sqlite3.connect('../data/geonames.db').cursor(), u"Bethlehem", pop_only=True))

# conn = sqlite3.connect('../data/geonames.db')
# c = conn.cursor()
# c.execute("INSERT INTO GEO VALUES (?, ?)", (u"darfur", u"[(13.5, 23.5, 0), (13.0, 25.0, 10000), (44.05135, -94.83804, 106)]"))
# c.execute("DELETE FROM GEO WHERE name = 'darfur'")
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

# c = []
# for line in codecs.open("/Users/milangritta/PycharmProjects/Research/data/lgl_edin.txt", "r", encoding="utf-8"):
#     if len(line.strip()) == 0:
#         continue
#     for toponym in line.split("||")[:-1]:
#         toponym = toponym.split(",,")
#         c.append(str((toponym[2], toponym[3])))
#
# c = Counter(c)
# counts = []
# for key in c.most_common():
#     counts.append(key[1])
# print(len(c)/4462.0)
#
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

# correlations = [x[0][1] for x in cPickle.load(open("data/correlations.pkl"))]
# correlations = [x[0][1] for x in correlations]
# minimum = min(correlations)
# ran = max(correlations) - minimum
# correlations = [x + ran for x in correlations]
# correlations = np.reshape(np.array(correlations), ((180 / GRID_SIZE), (360 / GRID_SIZE)))
# correlations = np.rot90((np.rot90(correlations)))
# visualise_2D_grid(correlations, "Geographical Pixel Embedding Quality")

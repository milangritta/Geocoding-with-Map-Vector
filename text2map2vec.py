# -*- coding: utf-8 -*-
import sqlite3
import cPickle
import numpy as np
import spacy

#######################################################################################
#                                                                                     #
# If you're only interested in using the loc2vec generation, I extracted the relevant #
# code into this python script for quick and dirty use. You still need at least the   #
# encoding map file (see ENCODING_MAP below). You also need a database 'geonames.db'  #
#                                                                                     #
#######################################################################################
ENCODING_MAP = cPickle.load(open(u"data/1x1_encode_map.pkl"))
OUTLIERS_MAP = cPickle.load(open(u"data/1x1_outliers_map.pkl"))
nlp = spacy.load(u'en')
conn = sqlite3.connect(u'../data/geonames.db').cursor()


def text2loc2vec(text, mapping, outliers, polygon_size):
    doc = nlp(text)
    location = u""
    entities = []
    for index, item in enumerate(doc):
        if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC"]:
            if not (item.ent_iob_ == u"B" and item.text.lower() == u"the"):
                location += item.text + u" "

        if location.strip() != u"" and (item.ent_type == 0 or index == len(doc) - 1):
            location = location.strip()
            coords = get_coordinates(conn, location)
            if len(coords) > 0:
                entities.extend(coords)
            location = u""

    entities = sorted(entities, key=lambda (a, b, c, d): c, reverse=True)
    loc2vec = np.zeros(len(mapping), )

    if len(entities) == 0:
        return loc2vec  # No locations? Return an empty vector.
    max_pop = entities[0][2] if entities[0][2] > 0 else 1
    for s in entities:
        index = coord_to_index((s[0], s[1]), polygon_size)
        if index in mapping:
            index = mapping[index]
        else:
            index = mapping[outliers[index]]
        loc2vec[index] += float(max(s[2], 1)) / max_pop
    return loc2vec / loc2vec.max() if loc2vec.max() > 0.0 else loc2vec


def coord_to_index(coordinates, polygon_size):
    """
    Convert coordinates into an array index. Use that to modify loc2vec polygon value.
    :param coordinates: (latitude, longitude) to compute
    :param polygon_size: integer size of the polygon? i.e. the resolution of the world
    :return: index pointing into loc2vec array
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


t = u"I was born in Ethiopia, then moved to the United States. I like to travel to London and Victoria as well."
l2v = text2loc2vec(text=t, mapping=ENCODING_MAP, outliers=OUTLIERS_MAP, polygon_size=1)
print(l2v)

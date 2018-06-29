# -*- coding: utf-8 -*-
import sqlite3
import cPickle
import numpy as np
import spacy

#######################################################################################
#                                                                                     #
# If you're only interested the Map Vector generation, I extracted the relevant       #
# code into this python script for quick and dirty use. You still need at least the   #
# encoding map files (see ENCODING_MAP below). You also need a database 'geonames.db' #
#                                                                                     #
#######################################################################################


def text2mapvec(doc, mapping, outliers, polygon_size, db):
    """
    Parse text, extract entities, create and return the MAP VECTOR.
    :param doc: the paragraph to turn into a Map Vector
    :param mapping: the map resolution file, determines the size of MAP VECTOR
    :param outliers: must be the same size/resolution as MAPPING
    :param polygon_size: the tile size must also match i.e. all three either 1x1 or 2x2, etc.
    :return: the map vector for this paragraph of text
    """
    location = u""
    entities = []
    for index, item in enumerate(doc):
        if item.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
            if not (item.ent_iob_ == u"B" and item.text.lower() == u"the"):
                location += item.text + u" "

        if location.strip() != u"" and (item.ent_type == 0 or index == len(doc) - 1):
            location = location.strip()
            coords = get_coordinates(db, location)
            if len(coords) > 0:
                entities.extend(coords)
            location = u""

    entities = sorted(entities, key=lambda (a, b, c, d): c, reverse=True)
    mapvec = np.zeros(len(mapping), )

    if len(entities) == 0:
        return mapvec  # No locations? Return an empty vector.
    max_pop = entities[0][2] if entities[0][2] > 0 else 1
    for s in entities:
        index = coord_to_index((s[0], s[1]), polygon_size)
        if index in mapping:
            index = mapping[index]
        else:
            index = mapping[outliers[index]]
        mapvec[index] += float(max(s[2], 1)) / max_pop
    return mapvec / mapvec.max() if mapvec.max() > 0.0 else mapvec


def coord_to_index(coordinates, polygon_size):
    """
    Convert coordinates into an array index. Use that to modify mapvec polygon value.
    :param coordinates: (latitude, longitude) to compute
    :param polygon_size: integer size of the polygon? i.e. the resolution of the world
    :return: index pointing into mapvec array
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


# ENCODING_MAP = cPickle.load(open(u"data/1x1_encode_map.pkl"))  # the resolution of the map
# OUTLIERS_MAP = cPickle.load(open(u"data/1x1_outliers_map.pkl"))  # dimensions must match the above
# nlp = spacy.load(u'en_core_web_lg')  # or spacy.load(u'en') depending on your Spacy Download (simple or full)
# conn = sqlite3.connect(u'../data/geonames.db').cursor()  # this DB can be downloaded using the GitHub link

# t = u"I was born in Ethiopia, then moved to the United States. I like to travel to London and Victoria as well."
# t = nlp(u"The Giza pyramid complex is an archaeological site on the Giza Plateau, on the outskirts of Cairo, Egypt.")
# map_vector = text2mapvec(doc=t, mapping=ENCODING_MAP, outliers=OUTLIERS_MAP, polygon_size=1, db=conn)
# print(map_vector)

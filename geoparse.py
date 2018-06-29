import sqlite3
import spacy
import numpy as np
from geopy.distance import great_circle
from keras.models import load_model
from preprocessing import index_to_coord, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1, get_coordinates, REVERSE_MAP_2x2
from text2map2vec import text2mapvec

model = load_model("../data/weights")
nlp = spacy.load(u'en_core_web_lg')  # or spacy.load(u'en') depending on your Spacy Download (simple or full)
conn = sqlite3.connect(u'../data/geonames.db').cursor()  # this DB can be downloaded using the GitHub link


def geoparse(text):
    doc = nlp(text)
    for entity in doc.ents:
        if entity.label_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
            name = entity.text

            # entity.start, entity.end, entity.start_char, entity.end_char
            # map_vector = text2mapvec(doc=nlp(subtext), mapping=ENCODING_MAP_1x1, outliers=OUTLIERS_MAP_1x1, polygon_size=1, db=conn)
            # prediction = model.predict([np.asarray(context_words), np.asarray(context_words), np.asarray(entities_strings),
            #                 np.asarray(entities_strings), np.asarray(map_vector), np.asarray(target_string)])
            prediction = index_to_coord(REVERSE_MAP_2x2[np.argmax(prediction)], 2)
            candidates = get_coordinates(conn, name)

            if len(candidates) == 0:
                print(u"Don't have an entry for", name, u"in GeoNames")
                raise Exception(u"Check your database, buddy :-)")

            max_pop = candidates[0][2]
            best_candidate = []
            bias = 0.9
            for candidate in candidates:
                err = great_circle(prediction, (float(candidate[0]), float(candidate[1]))).km
                best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * bias, (float(candidate[0]), float(candidate[1]))))
            best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]

            # England,, England,, 51.5,, -0.11,, 669,, 676 || - use evaluation script
            print name
            print "Coordinates:", best_candidate[1]



geoparse(u"Melbourne is a great venue for spotting a kangaroo or a koala.")

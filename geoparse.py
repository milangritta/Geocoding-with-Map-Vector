import cPickle
import codecs
import sqlite3
from genericpath import isfile
from os import listdir
import spacy
import numpy as np
from geopy.distance import great_circle
from keras.models import load_model
from preprocessing import index_to_coord, ENCODING_MAP_1x1, OUTLIERS_MAP_1x1, get_coordinates
from preprocessing import CONTEXT_LENGTH, pad_list, TARGET_LENGTH, UNKNOWN, REVERSE_MAP_2x2
from text2mapVec import text2mapvec

model = load_model("../data/weights")  # weights to be downloaded from Cambridge Uni repo, see GitHub.
nlp = spacy.load(u'en_core_web_lg')  # or spacy.load(u'en') depending on your Spacy Download (simple or full)
conn = sqlite3.connect(u'../data/geonames.db').cursor()  # this DB can be downloaded using the GitHub link
padding = nlp(u"0")[0]  # Do I need to explain? :-)
word_to_index = cPickle.load(open(u"data/words2index.pkl"))  # This is the vocabulary file

for word in nlp.Defaults.stop_words:  # This is only necessary if you use the full Spacy English model
    lex = nlp.vocab[word]             # so if you use spacy.load(u'en'), you can comment this out.
    lex.is_stop = True


def geoparse(text):
    """
    This function allows one to geoparse text i.e. extract toponyms (place names) and disambiguate to coordinates.
    :param text: to be parsed
    :return: currently only prints results to the screen, feel free to modify to your task
    """
    doc = nlp(text)  # NER with Spacy NER
    for entity in doc.ents:
        if entity.label_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
            name = entity.text if not entity.text.startswith('the') else entity.text[4:].strip()
            start = entity.start_char if not entity.text.startswith('the') else entity.start_char + 4
            end = entity.end_char
            near_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, entity.start - CONTEXT_LENGTH / 2):entity.start]], True, padding) + \
                       pad_list(CONTEXT_LENGTH / 2, [x for x in doc[entity.end: entity.end + CONTEXT_LENGTH / 2]], False, padding)
            far_inp = pad_list(CONTEXT_LENGTH / 2, [x for x in doc[max(0, entity.start - CONTEXT_LENGTH):max(0, entity.start - CONTEXT_LENGTH / 2)]], True, padding) + \
                      pad_list(CONTEXT_LENGTH / 2, [x for x in doc[entity.end + CONTEXT_LENGTH / 2: entity.end + CONTEXT_LENGTH]], False, padding)
            map_vector = text2mapvec(doc=near_inp + far_inp, mapping=ENCODING_MAP_1x1, outliers=OUTLIERS_MAP_1x1, polygon_size=1, db=conn, exclude=name)

            context_words, entities_strings = [], []
            target_string = pad_list(TARGET_LENGTH, [x.text.lower() for x in entity], True, u'0')
            target_string = [word_to_index[x] if x in word_to_index else word_to_index[UNKNOWN] for x in target_string]
            for words in [near_inp, far_inp]:
                for word in words:
                    if word.text.lower() in word_to_index:
                        vec = word_to_index[word.text.lower()]
                    else:
                        vec = word_to_index[UNKNOWN]
                    if word.ent_type_ in [u"GPE", u"FACILITY", u"LOC", u"FAC", u"LOCATION"]:
                        entities_strings.append(vec)
                        context_words.append(word_to_index[u'0'])
                    elif word.is_alpha and not word.is_stop:
                        context_words.append(vec)
                        entities_strings.append(word_to_index[u'0'])
                    else:
                        context_words.append(word_to_index[u'0'])
                        entities_strings.append(word_to_index[u'0'])

            prediction = model.predict([np.array([context_words]), np.array([context_words]), np.array([entities_strings]),
                                        np.array([entities_strings]), np.array([map_vector]), np.array([target_string])])
            prediction = index_to_coord(REVERSE_MAP_2x2[np.argmax(prediction[0])], 2)
            candidates = get_coordinates(conn, name)

            if len(candidates) == 0:
                print(u"Don't have an entry for", name, u"in GeoNames")
                continue

            max_pop = candidates[0][2]
            best_candidate = []
            bias = 0.905  # Tweak the parameter depending on the domain you're working with.
            # Less than 0.9 suitable for ambiguous text, more than 0.9 suitable for less ambiguous locations, see paper
            for candidate in candidates:
                err = great_circle(prediction, (float(candidate[0]), float(candidate[1]))).km
                best_candidate.append((err - (err * max(1, candidate[2]) / max(1, max_pop)) * bias, (float(candidate[0]), float(candidate[1]))))
            best_candidate = sorted(best_candidate, key=lambda (a, b): a)[0]

            # England,, England,, 51.5,, -0.11,, 669,, 676 || - use evaluation script to test correctness
            print name, start, end
            print u"Coordinates:", best_candidate[1]


# Example usage of the geoparse function below reading from a directory and parsing all files.
directory = u"/Users/milangritta/PycharmProjects/data/lgl/"
files = [f for f in listdir(directory) if isfile(directory + f)]
for f in files:
    for line in codecs.open(directory + f, encoding="utf-8"):
        print line
        geoparse(line)

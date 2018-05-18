import codecs
import random
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
import numpy
from geopy.distance import great_circle
from preprocessing import get_coordinates

# -------------------------------------------- ERROR CHECKING ----------------------------------------------

if False:
    """
    Check for XML formatting, duplicate articles, URLs, coordinate distances to Geonames database, 
    correct indexing of location names i.e. start and end character positions.
    """
    tree = ET.parse(u'data/GeoVirus.xml')
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    root = tree.getroot()
    duplicates = set()
    for article in root:
        text = article.find('text').text
        if text in duplicates:
            raise Exception(u'Duplicate titles/sources!')
        else:
            duplicates.add(text)
        for location in article.find('locations'):
            start = location.find('start').text
            end = location.find('end').text
            name = location.find('name').text
            url = location.find('page')
            if url.text != u"N/A":
                if url is None or not url.text.startswith(u"https://en.wikipedia.org/wiki/"):
                    raise Exception(u"URL corrupted!")
            chunk = text[int(start) - 1: int(end) - 1]
            if chunk != name:
                raise Exception(chunk + ", " + name)
            if location.find('altName') is not None:
                name = location.find('altName').text
            lat = location.find('lat').text
            lon = location.find('lon').text
            coords = get_coordinates(c, name)
            dist = 10000.0
            for coord in coords:
                gap = great_circle((float(lat), float(lon)), (coord[0], coord[1])).km
                if gap < dist:
                    dist = gap
            if dist > 1000:
                print u"Distance is large, please check if this is normal.", name, url, dist, lat, lon

# -------------------------------------------------- NUMBERS -------------------------------------------------------

if False:
    """
    Generate essential stats describing the nature of the dataset. Reported in the publication.
    """
    tree = ET.parse('data/GeoVirus.xml')
    root = tree.getroot()
    counter, continents, words, articles = [], 0, [], 0
    for article in root:
        articles += 1
        text = article.find("text").text
        words.extend(text.split())
        for location in article.findall("locations/location"):
            name = location.find("name")
            cont = location.find("continent")
            if cont is not None:
                continents += 1
            counter.append(name.text)
    print "Total Locations:", len(counter)
    counter = Counter(counter)
    print "Unique Locations:", len(counter)
    print "Most Common:", counter.most_common()
    print "Continents", continents
    counter = [j for (i, j) in counter.most_common()]
    print "Mean:", numpy.mean(counter)
    print "Median:", numpy.median(counter)
    print "Articles:", articles
    print "Total words:", len(words)


# ---------------------------------------------- GENERATION ------------------------------------------------

if False:
    """
    Before running the function, please create a directory called "geovirus" outside of the loc2vec directory.
    This function is used to convert the XML file into (1.) a directory of files where each file contains the
    text of each article i.e. 229 files will be created. (2.) a file "geovirus_gold.txt" containing the gold answers
    for each article. These two outputs will be used to generate evaluation files in preprocessing.py
    """
    tree = ET.parse(u"data/GeoVirus.xml")
    root = tree.getroot()
    f = codecs.open(u"data/geovirus_gold.txt", "w", "utf-8")
    c = 0
    counter = []
    for child in root:
        text = child.find('text').text
        gold_tops = []
        for location in child.findall('./locations/location'):
            start = location.find("start")
            end = location.find("end")
            name = location.find("name")
            if location.find('altName') is not None:
                alt_name = location.find('altName')
            else:
                alt_name = name
            counter.append(name.text)
            lat = location.find("lat")
            lon = location.find("lon")
            gold_tops.append(alt_name.text + ",," + name.text + ",," + lat.text + ",," + lon.text + ",," + start.text + ",," + end.text)
        for t in gold_tops:
            f.write(t + "||")
        f.write("\n")
        f_out = codecs.open(u"../data/geovirus/" + str(c), 'w', "utf-8")  # Files saved by numbers
        f_out.write(text)
        f_out.close()
        c += 1
    f.close()
    counter = Counter(counter)
    print counter.most_common()

# --------------------------------------SUBSAMPLING FOR INTER-ANNOTATOR AGREEMENT--------------------------------------

if False:
    """
    Generate a 10% random sample for the Inter Annotator Agreement figures.
    """
    iaa_check = codecs.open(u"data/iaa_check.txt", "w", "utf-8")
    iaa_test = codecs.open(u"data/iaa_test.txt", "w", "utf-8")
    tree = ET.parse(u'data/GeoVirus.xml')
    root = tree.getroot()

    for article in root:
        if random.randint(1, 10) > 9:
            text = article.find("text").text
            iaa_test.write("-------------NEW ARTICLE-----------------\n")
            iaa_test.write(text + "\n")
            print_count = 0
            for loc in article.findall("./locations/location"):
                print_count += 1
                start = int(loc.find("start").text)
                iaa_check.write(loc.find("page").text + "\n")
                iaa_check.write(loc.find("start").text + "\n")
                iaa_check.write(loc.find("name").text + "\n")
                if print_count <= 3:
                    iaa_test.write("-----------\n")
                    iaa_test.write("LOCATION NAME -> Asia\n")
                    iaa_test.write("LINK -> https://en.wikipedia.org/wiki/Asia\n")
                    iaa_test.write("START CHARACTER -> 100\n")

# -----------------------------------------COMPUTING INTER-ANNOTATOR AGREEMENT---------------------------------------

if False:
    """
    Compute IAA, print out overlaps and disagreements, then calculate IAA figures manually.
    """
    iaa_answers = []
    for index, line in enumerate(codecs.open(u"data/iaa_answers.txt", "r", "utf-8"), start=1):
        if index % 3 == 0:
            iaa_answers.append((url, start, line.strip()))
        elif index % 3 == 1:
            url = line.strip()
        else:
            start = int(line) + 1

    iaa_check = []
    for index, line in enumerate(codecs.open(u"data/iaa_check.txt", "r", "utf-8"), start=1):
        if index % 3 == 0:
            iaa_check.append((url, start, line.strip()))
        elif index % 3 == 1:
            url = line.strip()
        else:
            start = int(line)

    intersection = Counter(iaa_check) & Counter(iaa_answers)
    print len(intersection)
    check = Counter(iaa_check) - intersection
    answers = Counter(iaa_answers) - intersection
    iaa_check = list(check.elements())
    iaa_answers = list(answers.elements())
    print iaa_check
    print iaa_answers

# ----------------------------------------- END -------------------------------------------

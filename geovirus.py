import codecs
import random
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
from geopy.distance import great_circle
from preprocessing import get_coordinates

# --------------------------------------------ERROR CHECKING----------------------------------------------

if False:
    tree = ET.parse(u'data/GeoVirus.xml')
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    root = tree.getroot()
    duplicates = set()
    for article in root:
        text = article.find('text').text
        if text in duplicates:
            raise Exception('Duplicate titles/sources!')
        else:
            duplicates.add(text)
        for location in article.find('locations'):
            start = location.find('start').text
            end = location.find('end').text
            name = location.find('name').text
            url = location.find('page')
            if url is None:
                raise Exception("URL missing!")
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
            if dist > 500:
                print "AAARGRHG!!!!!", name, url, dist, lat, lon

                # COORDINATE LIMITS (180 x 360)!!!

                # tree.write('data/GeoVirusUpdated.xml')

# ----------------------------------------------STATISTICS------------------------------------------------


# tree = ET.parse('data/GeoVirus.xml')
# root = tree.getroot()
# for article in root:
#     for location in article:
#         print location.text


# ----------------------------------------------GENERATION------------------------------------------------

if False:
    """"""
    tree = ET.parse(u"data/GeoVirus.xml")
    root = tree.getroot()
    f = codecs.open(u"data/geovirus.txt", "w", "utf-8")
    c = 0
    counter = []
    for child in root:
        text = child.find('text').text
        gold_tops = []
        for location in child.findall('./locations/location'):
            # if location.find('continent') is not None:
            #     continue
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

if True:  # add CDATA?
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


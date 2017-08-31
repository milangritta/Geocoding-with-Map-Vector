import sqlite3
import xml.etree.ElementTree as ET
from geopy.distance import great_circle
from preprocessing import get_coordinates

# --------------------------------------------ERROR CHECKING----------------------------------------------

if True:  # add CDATA xml construct?
    tree = ET.parse(u'data/GeoVirus.xml')
    conn = sqlite3.connect(u'../data/geonames.db')
    c = conn.cursor()
    root = tree.getroot()
    for article in root:
        text = article.find('text').text
        for location in article.find('locations'):
            start = location.find('start').text
            end = location.find('end').text
            name = location.find('name').text
            url = location.find('page').text
            chunk = text[int(start) - 1: int(end) - 1]
            if chunk != name:
                print chunk, name
            lat = location.find('lat').text
            lon = location.find('lon').text
            coords = get_coordinates(c, name)
            dist = 10000.0
            for coord in coords:
                gap = great_circle((float(lat), float(lon)), (coord[0], coord[1])).km
                if gap < dist:
                    dist = gap
            if dist > 201:
                print "AAARGRHG!!!!!", name, url, dist

    # COORDINATE LIMITS (180 x 360)!!!

    # tree.write('data/GeoVirusUpdated.xml')

# ----------------------------------------------STATISTICS------------------------------------------------

# tree = ET.parse('data/GeoVirus.xml')
# root = tree.getroot()
# for article in root:
#     for location in article:
#         print location.text

# -----------------------------------------------ANALYSIS-------------------------------------------------


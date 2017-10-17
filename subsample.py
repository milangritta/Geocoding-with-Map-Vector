import codecs
import sqlite3
from geopy.distance import great_circle
from preprocessing import get_coordinates

counter = 0         # keeps track of current line number
start = 0           # where do you want to start sampling from?
finish = 1000000    # where do you want to end sampling?
frequency = 2       # 1 means take EVERY sample, 2 means take every SECOND sample, ...
output_file = u"../data/train_wiki_uniform.txt"
input_file = u"../data/train_wiki.txt"

filtering = True    # Do you want to filter samples with coordinate errors?
filtered_count = 0  # Keeping track of how many get filtered out
saved_count = 0     # Keeping track of how many samples were saved
max_distance = 5000 # The maximum size of the coordinate error (1 degree = 110km)
conn = sqlite3.connect(u'../data/geonames.db')
c = conn.cursor()   # Initialise database connection

out = codecs.open(output_file, u"w", encoding=u"utf-8")
for line in codecs.open(input_file, u"r", encoding=u"utf-8"):
    counter += 1
    if counter < start:
        continue
    if counter > finish:
        break
    if counter % frequency == 0:
        if not filtering:
            out.write(line)
            saved_count += 1
        else:
            split = line.split(u"\t")
            wiki_coordinates = (float(split[0]), float(split[1]))
            name = u" ".join(eval(split[5])).strip()
            db_coordinates = get_coordinates(c, name)
            distance = []
            for candidate in db_coordinates:
                distance.append(great_circle(wiki_coordinates, (float(candidate[0]), float(candidate[1]))).kilometers)
            distance = sorted(distance)
            if distance[0] > max_distance:
                print(name, distance[0])
                filtered_count += 1
            else:
                out.write(line)
                saved_count += 1

print(u"Saved", saved_count, u"samples.")
if filtering:
    print(u"Filtered:", filtered_count, u"samples.")

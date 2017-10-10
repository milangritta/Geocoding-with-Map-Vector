import sqlite3
from preprocessing import get_coordinates
conn = sqlite3.connect('../data/geonames.db')
c = conn.cursor()

print get_coordinates(c, u"")
c.execute(u"UPDATE GEO SET METADATA = '' WHERE NAME = ''")
conn.commit()
conn.close()
exit()

            # <location>
            #     <name></name>
            #     <start></start>
            #     <end></end>
            #     <lat></lat>
            #     <lon></lon>
            #     <page></page>
            #     <cont>1</cont>
            # </location>


# Which Melbourne? Augmenting Geocoding with Maps

### Resources accompanying the ACL 2018 publication.

Additional data required (files too large for GitHub), please download the following file -> *data.zip* <- from this LINK (coming soon). N.B. There is already a (small) data directory in this repository. Recording of the Melbourne presentation coming soon.

*The accepted pdf manuscript is also included in this directory (as is the .PPTX from the Melbourne presentation).*

#### Dependencies
* Keras 2.2.0
* Tensorflow 1.8
* Spacy 2.0 (also download a model https://spacy.io/usage/models)
* Also get sqlite, matplotlib, cpickle and geopy
* The rest should be installed alongside the three major libraries
* Next time I'll use Docker, too late now, sorry about that.

### Instructions
* Download the weights.zip and geonames.db.zip files (optional files available, see README).
* Read the README.md inside the directory to learn about the contents.
* Create a "data" directory outside the root directory to store the large files.
* Unzip the files into that directory, this will take up a few GBs of space.
* For replication, use **test.py**, see further instructions in the code.
* To tweak the model, use **train.py**, see comments inside the script.

For training, use a GPU, a single CPU epoch takes such a looooooong time, it's not worth it. Contact me on :envelope: *mg711 at cam dot ac dot uk* :envelope: if you need any help with reproduction or some other aspect of this work.

#### Tools
I included a couple of 'tools' for applied scientists and tinkerers in case you want to parse your own text and/or want to compare system performance with your research.
##### text2mapVec.py
This is a simple script that turns text into a Map Vector i.e. extracts locations/toponyms with Spacy NER and creates the 'bag of locations' or the Map Vector as an additional feature vector to be used in a downstream task.
##### geoparse.py
This is ...

*Disclaimer 1: We use **Spacy NER** for Named Entity Recognition. The reported F-Scores for each models can be found here: https://spacy.io/models/en It's not that great and will certainly affect performance. Use Oracle NER for a scientifically adequate comparison. Oracle means you extract the entities separately with perfect fidelity, then evaluate toponym recognition.*

*Disclaimer 2: The speed of execution won't be a record breaker, this is research code, I'm really busy trying to finish the PhD, sorry, I don't have time to rewrite it from scratch using proper software engineering principles. I hope you understand. Feel free to fork and edit.*

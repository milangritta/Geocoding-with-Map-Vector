# Which Melbourne? Augmenting Geocoding with Maps

### Code, Data and Resources accompanying the ACL 2018 publication.

Additional data is required, please download the following file -> *data.zip* <- from this LINK (coming soon). There is also a data directory in this repository (to avoid confusion plus the external data directory is too large for GitHub). *The accepted pdf manuscript is included in this directory (as is the PPTX from the Melbourne presentation).*

#### Dependencies
* Keras 2.2.0
* Tensorflow 1.8
* Spacy 2.0
* The rest should be installed alongside the three major libraries

### Instructions
* Unzip the *data.zip* file **outside** the project root directory to store the large files.
* Read the README.md inside the directory to learn about the contents.
* Unzip the necessary zipped files in the data directory (**minimum** geonames.db and weights), this will take up a few GBs of space.
* For replication, use **test.py**, see further instructions in the code.
* To tweak the model, use **train.py**, see comments inside the script.

For training, use a GPU, a single CPU epoch takes such a looooooong time, it's not worth it. Contact me on :envelope: *mg711 at cam dot ac dot uk* :envelope: if you need any help with reproduction or some aspect of this work.

#### Tools
I included a couple of tools for applied scientists and doers in case you want to parse your own text and/or want to compare system performance.
##### text2mapVec.py
This is a simple script that turns any text into a Map Vector i.e. extracts locations/toponyms with Spacy NER and creates the 'bag of locations' or the Map Vector as an additional feature vector to be used in a separate task.
##### geoparse.py
This is ... We use **Spacy NER** (use Oracle NER for accurate comparison).
*Disclaimer: The speed of execution won't be a record breaker, this is research code, I'm really busy trying to finish the PhD, sorry, I don't have time to rewrite it from scratch using proper software engineering principles. I hope you understand. Feel free to fork and edit.*

# Which Melbourne? Augmenting Geocoding with Maps

### Resources accompanying the ACL 2018 long paper, presented in Melbourne, Australia.

*The accepted pdf manuscript is also included in this directory (as is the .PPTX from the Melbourne presentation). Video recording of the Melbourne presentation coming soon.*

##### Abstract
The purpose of text geolocation is to associate geographic information contained
in a document with a set (or sets) of coordinates, either implicitly by using linguistic
features and/or explicitly by using geographic metadata combined with
heuristics. We introduce a geocoder (location mention disambiguator) that achieves
state-of-the-art (SOTA) results on three diverse datasets by exploiting the implicit
lexical clues. Moreover, we propose a new method for systematic encoding of
geographic metadata to generate two distinct views of the same text. To that end,
we introduce the Map Vector (MapVec), a sparse representation obtained by plotting
prior geographic probabilities, derived from population figures, on a World
Map. We then integrate the implicit (language) and explicit (map) features to significantly
improve a range of metrics. We also introduce an open-source dataset for
geoparsing of news events covering global disease outbreaks and epidemics to help
future evaluation in geoparsing.

##### Resources

This repository contains the accompanying data and source code for CamCoder (toponym resolver) described in the paper. Additional data is required as the files are too large for GitHub, please download files from this *<**LINK**>* (coming soon).

#### Dependencies
* Keras 2.2.0 https://keras.io/#installation
* Tensorflow 1.8 https://www.tensorflow.org/install/
* Spacy 2.0 (also download a model https://spacy.io/usage/models)
* Python 2.7+ and a recent version of sqlite, matplotlib, cpickle and geopy
* The rest should be installed alongside the three major libraries
* Next time I'll use Docker, too late now, sorry about that.

### Instructions
* Download the `weights.zip` and `geonames.db.zip` files as a **minimum** (optional files available from the Cambridge University <*DOWNLOAD LINK*> repo).
* Read the `README.txt` in the repository to learn about the contents.
* Create a **data** folder *outside the root directory* to store the large files. N.B. There is already a data folder **inside** the root directory! This holds the small files.
* Unzip the files into that directory, this will take up a few GBs of space.
* For replication, use `test.py` and see further instructions in the code. That should run out of the box if you followed the previous instructions. If not, get in touch!
* To tweak the model, use `train.py`, see comments inside the script for more info.

Use a GPU, if you can, a CPU epoch takes such a looooooong time, it's only worth it for small jobs. Contact me on :envelope: *mg711 at cam dot ac dot uk* :envelope: if you need any help with reproduction or some other aspect of this work at any time. After graduation, find me on Twitter/milangritta or raise an issue/ticket.

#### Tools
I included a couple of 'tools' for applied scientists and tinkerers in case you want to parse your own text and/or want to compare system performance with your research.
##### text2mapVec.py
This is a simple function `buildMapVec(text)` that turns text into a **Map Vector** i.e. extracts locations/toponyms with **Spacy NER** and creates the 'bag of locations' or the Map Vector as an additional feature vector to be used in a downstream task.

*NOTE: The speed of execution won't be a record breaker, this is research code, I'm really busy trying to finish the PhD, sorry, I don't have time to rewrite it from scratch using proper software engineering principles. I hope you understand. Feel free to fork and edit.*
##### geoparse.py
Unline most (maybe all) geoparsers, CamCoder can perform *geotagging* (NER) and *geocoding* separately. Use (1.) for the full pipeline and (2.) for toponym resolution only.
1. To geocode with NER: Use `geoparse(text)`, instructions in the code.
2. To geocode with Oracle: This will be slightly more laborious as you will need the `generate_evaluation_data(corpus, file_name)` function in `preprocessing.py`. First, save your evaluation dataset in the format of `data/lgl.txt` (name,,name,,lat,,lon,,start,end) then you don't have to modify any code. I think it's the best option. Once you have generated machine-readable data with that function, you're ready to `test.py` the performance.

*NOTE: CamCoder uses **Spacy NER** for Named Entity Recognition. The reported F-Scores for each model can be found here https://spacy.io/models/en, not that great and will certainly affect performance. Use **Oracle NER** for a scientifically adequate comparison. Oracle means you extract the entities separately with perfect fidelity, then evaluate toponym recognition in isolation. Also feel free to plug in a custom **NER tagger**, the code is extendable and should be well documented. Famous last words :-)*

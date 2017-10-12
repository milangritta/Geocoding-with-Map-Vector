# -*- coding: utf-8 -*-
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from preprocessing import generate_arrays_from_file_2D, GRID_SIZE, BATCH_SIZE
from subprocess import check_output

#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')

near_entities_coord = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
nec = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(near_entities_coord)
nec = MaxPooling2D(data_format="channels_first")(nec)
nec = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(nec)
nec = MaxPooling2D(data_format="channels_first")(nec)
nec = Flatten()(nec)
nec = Dropout(0.3)(nec)

far_entities_coord = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
fec = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(far_entities_coord)
fec = MaxPooling2D(data_format="channels_first")(fec)
fec = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(fec)
fec = MaxPooling2D(data_format="channels_first")(fec)
fec = Flatten()(fec)
fec = Dropout(0.3)(fec)

target_coord = Input(shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE))
tc = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(target_coord)
tc = MaxPooling2D(data_format="channels_first")(tc)
tc = Conv2D(10, (2, 2), activation="relu", input_shape=(1, 180 / GRID_SIZE, 360 / GRID_SIZE), data_format="channels_first")(tc)
tc = MaxPooling2D(data_format="channels_first")(tc)
tc = Flatten()(tc)
tc = Dropout(0.3)(tc)

inp = concatenate([nec, fec, tc])
inp = Dense(2000, activation="relu")(inp)
inp = Dense(units=(180 / GRID_SIZE) * (360 / GRID_SIZE), activation='softmax')(inp)
model = Model(inputs=[near_entities_coord, far_entities_coord, target_coord], outputs=[inp])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath="../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
# checkpoint = ModelCheckpoint(filepath="../data/weights", verbose=0)
early_stop = EarlyStopping(monitor='acc', patience=5)
file_name = u"../data/train_wiki_uniform.txt"
model.fit_generator(generate_arrays_from_file_2D(file_name),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=200, callbacks=[checkpoint, early_stop])

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from os import listdir
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from models import *
from generators import DataGeneratorCloudsData as DataGenerator

# Parameters samples
lags = 4
lat = 256
long = 256
feats = 1
ts_ahead = 1
shuffle_samples = True
continuity_days_samples = False


# Parameters network
convFilters = 16
dropoutRate = 0.5
dim_model = '3d'


# Parameters training
epochs = 40
batch_size = 8
metric = 'mse'
loss = 'binary_crossentropy'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


#Instantiation
# model = UNet_original(lags, lat, long, lags, feats, convFilters, dropoutRate)
# model = UNet_AsymmetricInceptionRes3DDR(lags, lat, long, feats, feats, convFilters, dropoutRate)
model = broad_UNet(lags, lat, long, feats, feats, convFilters, dropoutRate)
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
model.summary()


#Checkpoint to save best model
filepath="saved_models_clouds/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]


#Instantiating generators
percentage_validation_data = 0.2
try:
	training_generator = DataGenerator(dir_data='dataset_clouds/train', ts_lags=lags, ts_ahead=ts_ahead, batch_size=batch_size, shuffle=shuffle_samples, dim_model = dim_model)
except:
	raise Exception('\n\nNo data was found! Get and decompress the data as indicated first.')
validation_generator = training_generator.create_validation_generator(percentage_validation_data)


# Training
history = model.fit(training_generator, epochs=1, validation_data=validation_generator, callbacks=callbacks_list, use_multiprocessing=False)

#Showing training history
fig = plt.figure(figsize=(10,7))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title("Training history")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.ylim(top=0.005,bottom=0) #Limit
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.subplots_adjust(right=0.80, top=0.88)
plt.grid(b=None)
plt.savefig('training_results_clouds.png', dpi = 300)
plt.show()



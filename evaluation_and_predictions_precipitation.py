import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from models import *
from generators import DataGeneratorPrecipitationData as DataGenerator


##---- Evaluating model saved in "saved_models_precipitation/best_model.hdf5" ----##

# Select dataset
# filename = "dataset_precipitation/Data_20/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_20.h5"
filename = "dataset_precipitation/Data_50/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"


# Read dataset
try:
	f = h5py.File(filename, 'r')
except:
	raise Exception('\n\nNo data was found! Get and decompress the data as indicated first.')
    
# Test data to numpy array
data_test = f['/test/images']


#Parameters
lags = 12
lat = data_test.shape[-2]
long = data_test.shape[-1]
loss = 'mse'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
batch_size = 2


# Mean value training dataset 50% pixel occurence
threshold = 0.0013051119


# Custom metrics (Denormalized MSE and binarized metrics)
denormalized_mse = MSE_denormalized(47.83, batch_size, reduction_sum=True, latitude=lat, longitude=long) #MSE per image
binarized_metrics = thresholded_mask_metrics(threshold=threshold)
metrics = [denormalized_mse.mse_denormalized_per_image, denormalized_mse.mse_denormalized_per_pixel, binarized_metrics.acc, binarized_metrics.precision, binarized_metrics.recall, binarized_metrics.f1_score, binarized_metrics.CSI, binarized_metrics.FAR]


#Loade model and compile with custom metrics
filepath="saved_models_precipitation/best_model.hdf5"
try:
	model = load_model(filepath, compile=False)
except:
	raise Exception('\n\nNo trained model was found! Run first the trainig script or request pretrained model.')
model.compile(loss=denormalized_mse.mse_denormalized_per_image, optimizer=optimizer, metrics=metrics)


#Generator
test_generator = DataGenerator(data_test, batch_size, lags)


#Evaluate with generator 
print("\nEvaluating...")
result = model.evaluate(test_generator)

print("\n>>> Results evaluation:")
print(" - MSE:", result[2])
print(" - MSE per image:", result[1])
print(" - Acc:", result[3])
print(" - Precision:", result[4])
print(" - Recall:", result[5])



##---- Visualize individual prdiction examples ----##

for t in range(165, 170, 2):

    #Generating targets and labels
    x = data_test[t:(t+2), :lags, :, :]
    y = data_test[t:(t+2), -1, :, :]
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    y = np.expand_dims(y, axis=1)

    #Predicting
    pred = model.predict(x)

    #Visualizing predictions
    for i in range(len(pred)):     
        fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3.21, 4]})
        ax[0].imshow(y[i,0,:,:, 0], origin='lower')
        ax[0].set_title("Ground truth", fontsize=16)
        ax[0].axis('off')
        im=ax[1].imshow(pred[i,0,:,:,0], origin='lower') 
        ax[1].set_title("Prediction", fontsize=16)
        ax[1].axis('off')
        fig.tight_layout()
        fig.colorbar(im, shrink=0.71)
        plt.show()
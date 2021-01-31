import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
from models import *
from generators import DataGeneratorCloudsData as DataGenerator


##---- Evaluating model saved in "saved_models_precipitation/best_model.hdf5" ----##

# Parameters
ts_ahead = 1
loss = 'binary_crossentropy'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
lags = 4
batch_size = 8
shuffle_samples = True
dim_model = '3d'


# Metrics
binarized_metrics = thresholded_mask_metrics(threshold=0.5)
metrics =['mse', binarized_metrics.binarized_mse, binarized_metrics.acc, binarized_metrics.precision, binarized_metrics.recall] 


# Load model
filepath="saved_models_clouds/best_model.hdf5"
try:
	model = load_model(filepath, compile=False)
except:
	raise Exception('\n\nNo trained model was found! Run first the trainig script or request pretrained model.')
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# Evaluate
try:
	test_generator = DataGenerator(dir_data='dataset_clouds/test', ts_lags=lags, ts_ahead=ts_ahead, batch_size=batch_size, shuffle=shuffle_samples, dim_model = dim_model)
except:
	raise Exception('\n\nNo data was found! Get and decompress the data as indicated first.')
result = model.evaluate(test_generator)

print("\n>>> Results evaluation for",ts_ahead, "ts ahead:")
print(" - MSE:", result[1])
print(" - Binarized MSE:", result[2])
print(" - Acc:", result[3])
print(" - Precision:", result[4])
print(" - Recall:", result[5])



##---- Visualize individual prdiction examples ----##

for i in range(0, 1):

	# Predict one batch of samples
    x_test, y_test = test_generator.__getitem__(i)
    pred = model.predict(x_test)

    # Iterate over samples
    for j in range(len(x_test)):
        
    	# Binarize image
        rounded_pred = pred.copy()
        rounded_pred[rounded_pred>=0.5] = 1
        rounded_pred[rounded_pred<0.5] = 0

        # Visualize ground truth, prediction and binarized prediction
        fig, ax = plt.subplots(1,3, figsize=(10,5), gridspec_kw={'width_ratios': [3, 3, 3.76]})
        ax[0].set_title("Ground truth", fontsize=18)
        ax[0].imshow(y_test[j,0,:,:,0])
        ax[0].axis('off')
        ax[1].set_title("Prediction", fontsize=20)
        ax[1].imshow(pred[j,0,:,:,0])
        ax[1].axis('off')
        ax[2].set_title("Binarized prediction", fontsize=20)
        im=ax[2].imshow(rounded_pred[j,0,:,:,0])
        ax[2].axis('off')
        fig.tight_layout()
        cbr = fig.colorbar(im, shrink=0.62)
        cbr.ax.set_ylabel('Probability cloud cover', fontsize=17)
        cbr.ax.tick_params(labelsize=14)
        plt.show()
import h5py
import tensorflow as tf
import numpy as np


# Custom metric to calculate the denormalized for the precipitation data
class MSE_denormalized:
    def __init__(self, maxValue, batch_size, latitude, longitude, reduction_sum=False):
        self.maxValue = maxValue
        self.batch_size = batch_size
        self.n_pixels = tf.cast(tf.math.multiply(latitude, longitude), tf.float32)
        if reduction_sum:
            self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        else:
            self.mse = tf.keras.losses.MeanSquaredError()

    def mse_denormalized_per_image(self, y_true, y_pred):
        # Denormalizing ground truth
        y_true = tf.math.multiply(y_true, self.maxValue)
        # Denormalizing prediction
        y_pred = tf.math.multiply(y_pred, self.maxValue)
        # Calculating mse per image
        mse_image = tf.truediv(self.mse(y_true, y_pred), self.batch_size)
        return mse_image

    def mse_denormalized_per_pixel(self, y_true, y_pred):
        # Calculating mse per image
        mse_image = self.mse_denormalized_per_image(y_true, y_pred)
        # Calculating mse per pixel
        mse_pixel = tf.truediv(mse_image, self.n_pixels)
        return mse_pixel


# Convert output to binary mask and calculate metrics
class thresholded_mask_metrics:
    def __init__(self, test_data=None, threshold=None):
        self._binarized_mse = tf.keras.losses.MeanSquaredError()
        self._acc = tf.keras.metrics.Accuracy()
        self._precision = tf.keras.metrics.Precision()
        self._recall = tf.keras.metrics.Recall()
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = np.mean(test_data)
        self._TP = tf.keras.metrics.TruePositives()
        self._TN = tf.keras.metrics.TrueNegatives()
        self._FP = tf.keras.metrics.FalsePositives()
        self._FN = tf.keras.metrics.FalseNegatives()

    def binarize_mask(self, values):
        # Initialize TF values
        zero = tf.cast(tf.constant(0), tf.float32)
        one = tf.cast(tf.constant(1), tf.float32)
        limit = tf.cast(tf.constant(self._threshold), tf.float32)
        # Replacing values for 0s and 1s
        values = tf.where(tf.math.greater_equal(values, limit), one, values)
        values = tf.where(tf.math.less(values, limit), zero, values)
        return values
    
    def binarized_mse(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._binarized_mse(y_true, y_pred)

    def acc(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._acc(y_true, y_pred)

    def precision(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._precision(y_true, y_pred)

    def recall(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._recall(y_true, y_pred)

    def f1_score(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)

    def CSI(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate TP, TN, FP, FN
        TP = self._TP(y_true, y_pred)
        TN = self._TN(y_true, y_pred)
        FP = self._FP(y_true, y_pred)
        FN = self._FN(y_true, y_pred)
        # Calculate metrics
        return TP/(TP+FN+FP)

    def FAR(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate TP, TN, FP, FN
        TP = self._TP(y_true, y_pred)
        TN = self._TN(y_true, y_pred)
        FP = self._FP(y_true, y_pred)
        FN = self._FN(y_true, y_pred)
        # Calculate metrics
        return FP/(TP+FP)
      
        
def model_persistence(x):
    return x[:, :, -1]


def acc(y_true, y_pred):
    return np.sum((np.asarray(y_true) == np.asarray(y_pred))) / (y_true.shape[0]*y_true.shape[1])


def precision(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    return TP / (TP+FP)


def recall(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    return TP / (TP+FN) 


def extract_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
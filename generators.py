import tensorflow as tf
import numpy as np
from os import listdir



class DataGeneratorCloudsData(tf.keras.utils.Sequence):
    def __init__(self, dir_data, ts_lags, ts_ahead, batch_size, shuffle = False, dim_model = '3d'):    
        # Get path to samples
        self.path_samples = []
        self.dir_data = dir_data
        for s in listdir(dir_data):
            self.path_samples.append(dir_data+str('/')+s)
        self.path_samples = np.asarray(self.path_samples)
        
        # Shuffle data
        if shuffle:
            idx = np.random.permutation(len(self.path_samples))
            self.path_samples = self.path_samples[idx]
            
        self.b_size = batch_size
        self.num_samples = len(self.path_samples)
        self.ts_lags = ts_lags
        self.ts_ahead = ts_ahead
        self.dim_model = dim_model

      
    def create_validation_generator(self, percentage_data):
        # Instantiate validation generator
        val_generator = DataGeneratorCloudsData(dir_data=self.dir_data, ts_lags=self.ts_lags, ts_ahead=self.ts_ahead, batch_size=self.b_size, shuffle = False, dim_model=self.dim_model)
        # Split samples between training and validation
        val_generator.path_samples = self.path_samples[-int(self.num_samples*percentage_data):]
        val_generator.num_samples = len(val_generator.path_samples)
        self.path_samples = self.path_samples[:-int(self.num_samples*percentage_data)]
        self.num_samples = len(self.path_samples)
        return val_generator
        
        
    #Calculates the number of batches
    def __len__(self):
        return int(self.num_samples/self.b_size)

    
    #Obtains one batch of data 
    def __getitem__(self, idx):
        x = []
        y = []
      	
        # Reading images for samples
        for sample in self.path_samples[idx*self.b_size:(idx+1)*self.b_size]:
            x.append(np.load(sample)['arr_0'][:, :, :self.ts_lags])
            y.append(np.load(sample)['arr_0'][:, :, self.ts_lags+self.ts_ahead-1])
            
        # 3D_UNet
        if self.dim_model == '3d':
            x = np.moveaxis(x, -1, 1)
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
            y = np.moveaxis(y, -1, 1)
            y = np.expand_dims(y, axis=-1)
        # 2D_UNet
        else:
            y = np.expand_dims(y, axis=-1)

        x = np.asarray(x)
        y = np.asarray(y)
                
        return x, y



class DataGeneratorPrecipitationData(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, lags):
        self.data = data
        self.b_size = batch_size
        self.lags = lags
        self.time_steps = data.shape[0]

    
    #Calculates the number of batches: samples/batch_size
    def __len__(self):
        #Calculating the number of batches 
        return int(self.time_steps/self.b_size)

    
    #Obtains one batch of data 
    def __getitem__(self, idx):
        x = self.data[idx*self.b_size:(idx+1)*self.b_size, 0:self.lags, :, :]
        y = self.data[idx*self.b_size:(idx+1)*self.b_size, -1, :, :]
        
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        y = np.expand_dims(y, axis=1)
        
                
        return x, y
import numpy as np
import keras

class DataPreperator:
    """
    Prepares image data to be used across different training models
    
    This implementation is biased by how I've been preparing the data
    for specific models in other notebooks. Evolution of this API is warranted.
    """
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    val_x  = []
    val_y  = []
    
    def load_data(self):
        "Loads data from disk. Returns train_x, train_y, test_x, test_y"

        self.train_x = np.load('../data/serialized/train_x.npy')
        self.train_y = np.load('../data/serialized/train_y.npy')

        self.test_x  = np.load('../data/serialized/test_x.npy')
        self.test_y  = np.load('../data/serialized/test_y.npy')

        self.train_x = self.train_x.astype(np.float32)
        self.test_x  = self.test_x.astype(np.float32)
        
        self.test_x /= 255
        self.train_x /= 255

        # TODO: test a version with np.float16
        self.train_y = self.train_y.astype(np.float32)
        self.test_y  = self.test_y.astype(np.float32)
        
        n_classes = 2
        self.test_y = keras.utils.to_categorical([ label[0] for label in self.test_y ], n_classes)
        self.train_y = keras.utils.to_categorical([ label[0] for label in self.train_y ], n_classes)
        
        return (self.train_x, self.train_y, self.test_x, self.test_y)
        
    def print_counts(self):
        "output counts of train/test data to console"
        
        print("Train Inputs %s" % len(self.train_x))
        print("Train labels %s" % sum(self.train_y))
        print("Test Inputs %s" % len(self.test_x))
        print("Test labels %s" % sum(self.test_y))
        print("Validation Inputs %s" % len(self.val_x))
        print("Validation labels %s" % sum(self.val_y))
        
    def create_validation_split(self, split_percentage=0.5):
        """splits test data into validation set and test set. split_percentage is the
        proportion of data allocated to the validation set"""

        # NOTE: this works out to distribute game vs. not game labels pretty evenly
        rand_indices = np.random.permutation(len(self.test_x))
        val_size = round(len(self.test_x) * split_percentage)

        self.val_x = self.test_x[rand_indices[:val_size]]
        self.test_x = self.test_x[rand_indices[val_size:]]

        self.val_y = self.test_y[rand_indices[:val_size]]
        self.test_y = self.test_y[rand_indices[val_size:]]
        
        return (self.test_x, self.test_y, self.val_x, self.val_y)
    
    def get_data_sets(self):
        """return variables such that the callers can assign instance vars
        
           Returns (self.train_x, self.train_y, self.test_x, self.test_y, self.val_x, self.val_y)
        """
        return (self.train_x, self.train_y, self.test_x, self.test_y, self.val_x, self.val_y)

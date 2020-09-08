import numpy as np

class Normalizer:
    class FeatureScaler:
        """
            Feature scaling also known as Min-Max scaling, Min-Max normalization
            Default scaling in range <0, 1>
        """
        def __init__(self, scale=(0, 1)):
            self.a = scale[0]
            self.b = scale[1]                

        def transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")
            self.min = min(array)
            self.max = max(array)
            rv = [self.a + ((p - self.min) * (self.b - self.a))/(self.max - self.min) for p in array]
            return np.array(rv)
        
        def reverse_transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")
            if not (self.min or self.max):
                raise ValueError("Has to transform first")
            
            rv = [(((p - self.a)*(self.max - self.min))/(self.b - self.a)) + self.min for p in array]
            return np.array(rv)
    

    class ZScoreScaler:
        """
            Normalization using mean and standard deviation
            Works best for approximately normally distributed data
        """
        def __init__(self):
                pass
        
        def transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")    

            self.mean = array.mean()
            self.std = np.std(array)
            rv = [(p - self.mean)/self.std for p in array]
            return np.array(rv)

        def reverse_transform(self, array: np.ndarray) -> np.ndarray: 
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")

            if not (self.mean or self.std):
                raise ValueError("Has to transform first")

            rv = [((p*self.std) + self.mean) for p in array]
            return np.array(rv)
    

    class MeanScaler:
        """
            Normalization using mean
        """
        def __init__(self):
                pass
        
        def transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")    

            self.mean = array.mean()
            self.min = min(array)
            self.max = max(array)
            rv = [(p - self.mean)/(self.max - self.min) for p in array]
            return np.array(rv)

        def reverse_transform(self, array: np.ndarray) -> np.ndarray: 
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")
    

    class UnitLengthScaler:
        """
            Normalization by Euclidean length
        """
        def transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")  
            
            self.euclidean_length = np.linalg.norm(array)
            rv = [p / self.euclidean_length for p in array]
            return np.array(rv)
        
        def reverse_transform(self, array: np.ndarray) -> np.ndarray:
            if not isinstance(array, np.ndarray):
                raise ValueError("Array has to be numpy array")  
            if not self.euclidean_length:
                raise ValueError("Has to transform first")

            rv = [p*self.euclidean_length for p in array]
            return np.array(rv)

    def __init__(self):
        self.FeatureScaler = self.FeatureScaler()
        self.MinMaxScaler = self.FeatureScaler()
        self.MeanScaler = self.MeanScaler()
        self.ZScoreScaler = self.ZScoreScaler()
        self.UnitLengthScaler = self.UnitLengthScaler()


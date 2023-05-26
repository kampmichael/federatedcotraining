import numpy as np

class Parameters:
    '''
    Super class of the different model- and DL library-dependent model parameter classes.
    '''

    def __init__(self):
        '''

        Implementations of this method initialize an object of 'Parameters' sub-class.

        Parameters
        ----------
        param

        Exception
        ---------
        ValueError
        '''

        raise  NotImplementedError

    def set(self):
        '''

        Implementations of this method set the model parameters.

        Parameters
        ----------
        weights

        Returns
        -------

        '''

        raise NotImplementedError

    def get(self):
        '''

        Implementations of this method return the current model parameters.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def add(self, other):
        '''

        Implementations of this method add model parameters of two models.

        Parameters
        ----------
        other: object - the model parameters to be added to the current model parameters

        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def scalarMultiply(self, scalar):
        '''

        Implementations of this method multiply the model parameters element-wise by a scalar.

        Parameters
        ----------
        scalar: float - the multiplication factor

        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def distance(self, other):
        '''

        Implementations of this method calculate the (e.g. Euclidian) distance between the current and another model in model parameter space.

        Parameters
        ----------
        other: object - the model parameters to be added to the current model parameters

        Returns
        -------
        
        '''
         
        raise NotImplementedError

    def getCopy(self):
        '''

        Implementations of this method return a copy of the current model parameters.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def toVector(self) -> np.array:
        '''

        Implementations of this method returns the current model parameters as a 1D numpy array.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def fromVector(self, v : np.array):
        '''

        Implementations of this method sets the current model parameters to the values given in the 1D numpy array v.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
        
class LinearParameters(Parameters):
    '''
    Specific implementation of Parameters class for PyTorchNN learner
    Here we know that parameters are list of numpy arrays. All the methods 
    for addition, multiplication by scalar, flattening and finding distance
    are contained in this class.

    '''
    def __init__(self, weights : np.array):
        self.weights = weights

    def set(self, weights : np.array):
        self.weights = weights
        # to use it inline
        return self

    def get(self):
        return self.weights
    
    def add(self, other : np.array):
        if not isinstance(other, LinearParameters):
            error_text = "The argument other is not of type" + str(LinearParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if otherW.shape != self.weights.shape:
            raise ValueError("Error in addition: weights have different shape. This: "+str(self.weights.shape)+", other: "+str(otherW.shape)+".")
        
        self.weights = np.add(self.weights, otherW)
        return self  # to use it inline
    
    def scalarMultiply(self, scalar):
        self.weights *= scalar
        return self  # to use it inline
    
    def distance(self, other : np.array):
        if not isinstance(other, LinearParameters):
            error_text = "The argument other is not of type" + str(LinearParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if otherW.shape != self.weights.shape:
            raise ValueError("Error in addition: weights have different shape. This: "+str(self.weights.shape)+", other: "+str(otherW.shape)+".")
        
        dist = np.linalg.norm(self.weights-otherW)
        
        return dist

    def getCopy(self):
        return LinearParameters(np.copy(self.weights))
    
    def toVector(self) -> np.array:
        return self.weights
    
    def fromVector(self, v : np.array):
        self.weights = v

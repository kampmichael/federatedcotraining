from typing import List
from parameters import Parameters
import numpy as np


class Aggregator():


    def __init__(self, name = "Aggregator"):
        self.name = name

    def __call__(self, params : List[Parameters]) -> Parameters:
        '''
        Aggregator call method, combines Parameters into one model's Parameters
        Specific implementation is different for different approaches.

        Parameters
        ----------
        params - list with Parameters of models to be aggregated

        Returns
        -------
        Parameters object for the aggregated model

        '''

        raise NotImplementedError
        
class Average(Aggregator):
    '''
    Provides a method to calculate an averaged model from n individual models (using the arithmetic mean)
    '''

    def __init__(self, name = "Average"):
        '''

        Returns
        -------
        None
        '''
        Aggregator.__init__(self, name = name)

    def calculateDivergence(self, param1, param2):
        if type(param1) is np.ndarray:
            return np.linalg.norm(param1 - param2)**2
        else:
            return param1.distance(param2)**2

    def __call__(self, params : List[Parameters]) -> Parameters:
        '''

        This aggregator takes n lists of model parameters and returns a list of component-wise arithmetic means.

        Parameters
        ----------
        params A list of Paramters objects. These objects support addition and scalar multiplication.

        Returns
        -------
        A new parameter object that is the average of params.

        '''
        newParams = params[0].getCopy()
        for i in range(1,len(params)):
            newParams.add(params[i])
        newParams.scalarMultiply(1/float(len(params)))
        #newParams = np.array(newParams)
        return newParams
        
    def __str__(self):
        return "Averaging"
        
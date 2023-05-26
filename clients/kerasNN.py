from parameters_keras import KerasNNParameters 
import numpy as np
import clients
from clients.papernet import SimpleCifar10NetKeras,Cifar10paperNetKeras
import tensorflow as tf
from tensorflow import keras 
from clients import *
from typing import List


class KerasNN(Client):
    def __init__(self):
        import tensorflow as tf
        super(Client, self).__init__()
        self.name = "KerasNN base class"

    def setCore(self, network):
        self._core = network

    def setModel(self, param: KerasNNParameters, setReference: bool):
        super(KerasNN, self).setModel(param, setReference)
        if setReference:
            self._flattenReferenceParams = self._flattenParameters(param)

    def checkLocalConditionHolds(self) -> (float, bool):
        '''

        Calculates the divergence of the local model from the reference model (currentDivergence) and compares it with the pre-defined divergence threshold (_delta)

        Returns
        -------
        bool

        '''
        localConditionHolds = True
        self._syncCounter += 1
        if self._syncCounter == self._syncPeriod:
            msg, localConditionHolds = self._synchronizer.evaluateLocal(self._flattenParameters(self.getParameters()), self._flattenReferenceParams)
            self._syncCounter = 0

        return msg, localConditionHolds

    def train(self, X, y) -> List:
        '''

        Calls the keras method "train_on_batch" that performs a single gradient update of the model based on batch "data" and returns performance of that updated model on "data"

        Parameters
        ----------
        data

        Returns
        -------
        scalar training loss

        Exception
        ---------
        AttributeError
            in case core is not set
        ValueError
            in case that data is not an numpy array
        '''
        if self._core is None:
            self.error("No core is set")
            raise AttributeError("No core is set")


        #self.info('STARTTIME_train_on_batch: '+str(time.time()))
        #with self._session.as_default():
        #    with self._session.graph.as_default():
        metrics = self._core.train_on_batch(X, y)
        #self.info('ENDTIME_train_on_batch: '+str(time.time()))
        return metrics
    
    def predict(self, X) -> np.ndarray:
        from tensorflow.keras.utils import to_categorical
        output = self._core.predict(X)
        output = to_categorical(np.argmax(output, axis=1), output.shape[1])
        return output
        
    def setParameters(self, param : KerasNNParameters):
        '''

        Replace the current values of the model parameters with the values of "param"

        Parameters
        ----------
        param

        Returns
        -------

        Exception
        ---------
        ValueError
            in case that param is not of type Parameters
        '''

        if not isinstance(param, KerasNNParameters):
            error_text = "The argument param is not of type" + str(KerasNNParameters) + "it is of type " + str(type(param))
            self.error(error_text)
            raise ValueError(error_text)

        #with self._session.as_default():
        #    with self._session.graph.as_default():
        self._core.set_weights(param.get())

    def getParameters(self) -> KerasNNParameters:
        '''

        Takes the current model parameters and hands them to a KerasNNParameters object which is returned

        Returns
        -------
        Parameters

        '''

        #with self._session.as_default():
        #    with self._session.graph.as_default():
        return KerasNNParameters(self._core.get_weights())
    
    def _flattenParameters(self, param):
        flatParam = []
        for wi in param.get():
            flatParam += np.ravel(wi).tolist()
        return np.asarray(flatParam)

@is_client        
class PaperKerasCIFARNet(KerasNN):
    def __init__(self):
        super(PaperKerasCIFARNet)
        self.name = "PaperKerasCIFARNet"
        learningRate = 0.01
        #updateRule = 'adam'
        updateRule = keras.optimizers.Adam(learning_rate=learningRate)
        lossFunction = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics=['accuracy']
        nn = Cifar10paperNetKeras()
        #nn.summary()
        nn.compile(loss=lossFunction, optimizer=updateRule, metrics = metrics)
        #nn.metrics_tensors += nn.outputs
        self.setCore(nn)
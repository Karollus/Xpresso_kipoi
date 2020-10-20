from kipoi.model import BaseModel
from keras.models import load_model
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

class Xpresso(BaseModel):
        def __init__(self, weights):
            self.nuc_dict = {'A':[1 ,0 ,0 ,0 ],'C':[0 ,1 ,0 ,0 ],'G':[0 ,0 ,1 ,0 ],
                             'T':[0 ,0 ,0 ,1 ],'N':[0 ,0 ,0 ,0 ]]}
            self.weights = weights
            self.model = load_model(weights)

        # One-hot encodes a particular sequence
        def encode_seq(self, seq):
            # Add padding:
            if (len(seq) != 10500):
                sys.exit( "Error in sequence %s: length is not equal to the required 10,500 nts. \
                Please fix or pad with Ns if necessary, with intended TSS centered at position 7,000." % seq )
            seq = seq.upper() # force upper case to be sure!
            # One hot encode
            try:
                one_hot = np.array([self.nuc_dict[x] for x in seq]) # get stacked on top of each other
            except KeyError as e:
                raise ValueError('Cant one-hot encode unknown base: {} in seq: {}. \
                                 Must be A, C, G, T, or N. If so, please filter'.format(str(e), seq))
            return one_hot

        # One-hot encodes the entire tensor
        def encode(self, inputs):
            # One Hot Encode input
            one_hot = np.stack([self.encode_seq(seq)
                                for seq in inputs], axis = 0)
            return one_hot

        # Predicts for a batch of inputs
        def predict_on_batch(self, inputs):
            # Encode
            one_hot = self.encode(inputs)
            #In this limited model, treat RNA as having average mRNA features, to ignore half-life contribution
            #For full model with half-life features, see Xpresso Github
            mean_half_life_features = np.zeros((inputs.shape[0],6), dtype='float32')
            pred = self.model.predict_on_batch([one_hot, mean_half_life_features]).reshape(-1)
            return {"expression_pred": pred}

import scipy.io as sio
import numpy as np


class ReadFiles(object):
    def __init__(self):
        spamData = sio.loadmat('../data/spam_data.mat', struct_as_record=False)

        self.header = spamData['__header__']
        self.version = spamData['__version__']
        self.names = spamData['names']

        pTrain = spamData['P_train']
        pTest = spamData['P_test']
        self.features = np.concatenate((pTrain, pTest), axis=1)
        self.features = self.features.transpose()
        self.log("Features Matrix Created and Imported")

        tTest = spamData['T_test']
        tTrain = spamData['T_train']
        self.labels = np.concatenate((tTrain, tTest), axis=1)
        self.labels = self.labels.transpose()
        self.labels = np.ravel(self.labels)
        self.log("Labels Array Created and Imported")

    def getFeatures(self):
        return self.features

    def getLabels(self):
        return self.labels

    def log(self, msg):
        print('[Reading Files] {}'.format(msg))

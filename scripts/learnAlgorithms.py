from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# import numpy as np




class LearnAlgorithms(object):

    def __init__(self, data, labels):
        self.log("Learning Initialized")
        self.pTrain, self.pTest, self.tTrain, self.tTest = train_test_split(data, labels,
                                                                            test_size=0.10)

    def runMLP(self):
        self.log("Running MLP...")
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(self.pTrain, self.tTrain)
        pred = cross_val_predict(rf, self.images, self.labels, cv=self.cv)



    def runFBR(self):
        self.log("Running FBR...")
        clf = RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0)

    def log(self, msg):
        print('[Learn] {}'.format(msg))

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict

class LearnAlgorithms(object):

    def __init__(self, data, labels):
        self.log("Learning Initialized")
        self.data = data
        self.labels = labels

    def runMLP(self):
        self.log("Running MLP...")
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        cv = ShuffleSplit(n_splits=20, test_size=0.1)
        score = cross_val_score(clf, self.data, self.labels, cv=cv)
        log(score)

    def runFBR(self):
        self.log("Running FBR...")
        # clf = RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))

    def log(self, msg):
        print('[Learn] {}'.format(msg))

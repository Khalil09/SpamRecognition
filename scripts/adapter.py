import readfiles as rf
import learnAlgorithms as learn

class Adapter(object):

    def __init__(self, algorithm):
        reader = rf.ReadFiles()
        self.features = reader.getFeatures()
        self.labels = reader.getLabels()
        self.la = learn.LearnAlgorithms(self.features, self.labels)
        self.algorithm = algorithm


    def run(self):
        self.log("Running...")
        if (self.algorithm == 'mlp'):
            self.la.runMLP()

    def log(self, msg):
        print('[Adapter] {}'.format(msg))

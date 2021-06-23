import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = []
        for i in np.arange(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def author(self):
        return 'kgao47'
    def add_evidence(self,data_x,data_y):
        for i in np.arange(20):
            self.learners[i].add_evidence(data_x,data_y)
    def query(self, points):
        pred_all = []
        for i in np.arange(20):
            pred_all.append(self.learners[i].query(points))
        return np.mean(np.array(pred_all), axis=0)
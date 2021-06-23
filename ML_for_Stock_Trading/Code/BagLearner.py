"""
Bag Learner
Author: Kun Gao (GT ID: 903612738)
How to use:
    import BagLearner as bl
    learner = bl.BagLearner(learner = al.ArbitraryLearner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False)
    learner.add_evidence(Xtrain, Ytrain)
    Y = learner.query(Xtest)
"""
import numpy as np
from scipy import stats

class BagLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):

        """
        Hints from project description:
        learners = []
        kwargs = {"k": 10}
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        """

        self.learners = []
        for i in np.arange(bags):
            self.learners.append(learner(**kwargs))

        self.bags = bags 
        self.verbose = verbose
        self.boost = boost

        if self.verbose and self.boost:
            print('Note: boosting is not supported')
        pass
			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'kgao47'
	 		  		  		    	 		 		   		 		  
    def addEvidence(self,data_x,data_y):
        nrec = np.shape(data_x)[0]
        for i in np.arange(self.bags):
            index_sel = np.random.choice(nrec, nrec, replace=True) # some records repeated
            data_x_sel = data_x[index_sel, :]
            data_y_sel = data_y[index_sel]
            if self.verbose:
                print('--- selecting a random portion of training dataset for bag', i)
                print('index:', index_sel)
                print('shape of data_x:', np.shape(data_x_sel))
                print('shape of data_y:', np.shape(data_y_sel))
            self.learners[i].addEvidence(data_x_sel,data_y_sel)

    def query(self, points):
        pred_all = []
        for i in np.arange(self.bags):
            pred_all.append(self.learners[i].query(points))
        pred_all = np.array(pred_all)
        #pred = np.mean(pred_all, axis=0) 
        pred = stats.mode(pred_all)[0][0] #use mode for classification
        return pred

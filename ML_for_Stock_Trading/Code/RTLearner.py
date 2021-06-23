"""
Random Tree Learner
Author: Kun Gao (GT ID: 903612738)
How to use:
    import RTLearner as rt
    learner = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
    learner.add_evidence(Xtrain, Ytrain) # training step
    Y = learner.query(Xtest) # query
"""

import numpy as np
from scipy import stats

  		   	  			  	 		  		  		    	 		 		   		 		  
class RTLearner(object):
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size	
        self.verbose = verbose
        if self.verbose:
            print('You are using the Random Tree Learner')
            print('The leaf size is', self.leaf_size)
  			  	 		  		  		    	 		 		   		 		  
    def author(self):
        return "kgao47"

    @staticmethod
    def find_feature_idx(x):
        """
        Find a random index in x
        """
        _, total_idx = np.shape(x)
        return np.random.randint(0,total_idx)

    def addEvidence(self, data_x, data_y):
        """
        Add training data to learner
        data_x: A set of feature values used to train the learner
        data_y: The value we are attempting to predict given the X data
        """
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        """
        Build a decision tree based on the algorithm in Balch slides
        Return: a 2-dim array describing the tree
        Note: the number -999 is used to represent a 'leaf' node
        """
        if data_x.shape[0] <= self.leaf_size: # if all data can fit in the same leaf
            #return np.array([[-999, np.mean(data_y), np.nan, np.nan]])
            return np.array([[-999, stats.mode(data_y,axis=None)[0][0], np.nan, np.nan]])
        elif np.max(data_y) == np.min(data_y): # if all labels are the same
            return np.array([[-999, data_y[0], np.nan, np.nan]])
        else:
            # find a random feature i to split on
            idx = self.find_feature_idx(data_x)
            SplitVal = np.median(data_x[:,idx])

            data_x_left = data_x[data_x[:,idx] <= SplitVal]
            data_y_left = data_y[data_x[:,idx] <= SplitVal]
            data_x_right = data_x[data_x[:,idx] > SplitVal]
            data_y_right = data_y[data_x[:,idx] > SplitVal]

            if len(data_y_right) == 0: # all data on the same side
                #return np.array([[-999, np.mean(data_y_left), np.nan, np.nan]])
                return np.array([[-999, stats.mode(data_y_left,axis=None)[0][0], np.nan, np.nan]])
            else: # ok, needs recursion
                left = self.build_tree(data_x_left, data_y_left)
                right= self.build_tree(data_x_right,data_y_right)
                root = np.array([[idx, SplitVal, 1, np.shape(left)[0]+1]])
                combined = np.vstack((root, left, right))
                return combined

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        points: a numpy array with each row corresponding to a specific query.
        return: the predicted result of the input data according to the trained model
        """
        nrec,_ = np.shape(points)
        pred_all = np.zeros(nrec) # output

        for i in np.arange(nrec):
            one_entry = points[i,:]
            ti = 0
            while self.tree[ti][0] > -900: # if not a leaf
                if one_entry[np.int(self.tree[ti][0])] <= self.tree[ti][1]:
                    ti += np.int(self.tree[ti][2]) # left
                else:
                    ti += np.int(self.tree[ti][3]) # right
            pred_all[i] = self.tree[ti][1]

        return pred_all

if __name__ == "__main__":
    print("Main")
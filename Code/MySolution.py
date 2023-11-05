import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
import cvxpy as cp
import copy

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None
        
        # K SVM's ? W = [w_1, w_2, ..., w_k]
        # objective: min()
        self.SVM_table = {}
        self.iteration = 100
        self.lambda_list = np.arange(0.1, 2.0, 0.5) # Test acc : 0.926, Train Acc: 0.918; Test Acc: 0.878, Train Acc: 0.976
        # self.lambda_list = [10.0] # Test Acc: 0.924, Train Acc: 0.914

    def train_single_svm(self, indices, trainX, trainY):
        # trainX: (b, f)
        # trainY: (b, 1)
        print(f"Shape of X: {trainX.shape}, shape of Y: {trainY.shape}")
        i, j = indices
        trainY_cur = copy.deepcopy(trainY)
        trainY_cur = np.logical_or(trainY_cur == i, trainY_cur == j)
        trainX_cur = trainX[trainY_cur.reshape(-1), :]
        trainY_ij = (trainY == i).astype(np.int8) # put 1 on label == i, 0 elsewhere
        trainY_ij = trainY_ij * 2 # put 2 on label == i, 0 elsewhere
        trainY_ij = trainY_ij - 1 # put 1 on label == i, -1 elsewhere

        # Sanity check of slice of X and y
        # trainY_ij[~trainY_cur.reshape(-1)] = 0
        # print(trainY[:10])
        # print(trainY_ij[:10])

        trainY_cur = trainY_ij[trainY_cur.reshape(-1)] # only keep entries where the original label == i or == j
        trainY_cur = trainY_cur.astype(np.int8)
        batch_size, feature_dim = trainX_cur.shape
        print(batch_size, feature_dim)

        for lambd in self.lambda_list:
            w = cp.Variable(feature_dim, 'w')
            b = cp.Variable(1, 'b')

            auxiliary_variables_for_L1_norm = cp.Variable(feature_dim, 'aux1')
            constraint_for_auxiliary_variable = [
                -auxiliary_variables_for_L1_norm <= w,
                w <= auxiliary_variables_for_L1_norm,
            ]

            auxiliary_variables_for_huber_loss = cp.Variable(batch_size, 'aux2')
            constraint_for_huber_loss = [
                auxiliary_variables_for_huber_loss >= 0,
                auxiliary_variables_for_huber_loss >= 1 - cp.multiply(trainY_cur, trainX_cur @ w + b)
            ]
            
            huber_loss = cp.sum(auxiliary_variables_for_huber_loss)
            total_constraint = constraint_for_huber_loss + constraint_for_auxiliary_variable

            prob = cp.Problem(
                # cp.Minimize(huber_loss/batch_size + cp.sum(auxiliary_variables_for_L1_norm)),
                cp.Minimize(huber_loss + lambd * cp.sum(auxiliary_variables_for_L1_norm)),
                total_constraint
            )

            prob.solve()
            self.SVM_table[indices+(lambd, )] = (w.value, b.value)

            print('====== Finish Solving SVM ======')
            print(prob)
            for variable in prob.variables():
                print("Variable %s: value %s" % (variable.name(), variable.value))

            # objective = <x, w/||w||> - y

    def predict_single(self, indices, trainX, trainY):
        i, j = indices
        trainY_cur = copy.deepcopy(trainY)
        trainY_cur = np.logical_or(trainY_cur == i, trainY_cur == j)
        trainX_cur = trainX[trainY_cur.reshape(-1), :]
        trainY_ij = (trainY == i).astype(np.int8) # put 1 on label == i, 0 elsewhere
        trainY_ij = trainY_ij * 2 # put 2 on label == i, 0 elsewhere
        trainY_ij = trainY_ij - 1 # put 1 on label == i, -1 elsewhere
        trainY_cur = trainY_ij[trainY_cur.reshape(-1)] # only keep entries where the original label == i or == j
        trainY_cur = trainY_cur.astype(np.int8)

        svm = self.SVM_table[indices]        

        logits = trainX_cur @ svm[0] + svm[1]
        y_hat = logits > 0
        y_hat = logits.astype(np.int8)
        y_hat = logits * 2 - 1

        acc = y_hat == trainY_cur
        acc = acc.sum() / trainY_cur.shape[0]
        print(f"{(i, j)}-th SVM classification accuracy is {acc}")

        return y_hat
    
    def train(self, trainX, trainY):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        indices = np.unique(trainY)
        indices = np.sort(indices)
        self.indices = indices
        for i in indices:
            for j in indices:
                if j <= i: continue
                self.train_single_svm((i, j), trainX, trainY)
      
    
    def predict(self, testX):
        ''' Task 1-2
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''
        predY = []
        for x_index in range(testX.shape[0]):
            x = testX[x_index, :].reshape(1, -1)
            vote = {}
            for index in self.indices:
                vote[index] = 0
            for indices in self.SVM_table.keys():
                for lambd in self.lambda_list:
                    i, j, _ = indices
                    svm = self.SVM_table[indices]
                    logits = x @ svm[0] + svm[1]
                    y_hat = logits > 0
                    y_hat = logits.astype(np.int8)
                    i_vote = y_hat.sum()
                    j_vote = 1 - i_vote
                    vote[i] += i_vote
                    vote[j] += j_vote
            # final_decision = np.argmax(vote)
            vote_list = [(index, vote[index]) for index in self.indices]
            vote_list.sort(key=lambda x : x[1], reverse=True)
            final_decision = vote_list[0][0]

            predY.append(final_decision)

        predY = np.array(predY)
        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

if __name__ == '__main__':
    from utils import *
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    data = prepare_synthetic_data()
    # data = prepare_mnist_data()
    svms = MyClassifier(K=3)
    # svms.train_single_svm((1, 2), data['trainX'], data['trainY'])
    # new_trainX = pca.fit_transform(data['trainX'])
    # new_testX = pca.transform(data['testX'])
    # svms.train(data['trainX'], data['trainY'])
    svms.train(data['trainX'], data['trainY'])
    # pred = svms.predict(data['trainX'])
    # print(pred)
    acc = svms.evaluate(data['testX'], data['testY'])
    acc2 = svms.evaluate(data['trainX'], data['trainY'])
    # acc = svms.evaluate(data['testX'], data['testY'])
    # acc2 = svms.evaluate(data['trainX'], data['trainY'])
    print(f"Final Test Prediction Acc: {acc}, Train Acc: {acc2}")

    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''


        # Update and teturn the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        pred_labels = None

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self,trainY):
        label_reference = self.align_cluster_labels(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_data_labels(testX)
        label_reference = self.align_cluster_labels(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        data_to_label = None

        # Return an index list that specifies which data points to label
        return data_to_label

    
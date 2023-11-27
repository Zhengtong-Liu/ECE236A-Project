import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
### TODO: import any other packages you need for your solution
import cvxpy as cp
import copy
import random

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K, ensemble=False):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None
        
        # K SVM's ? W = [w_1, w_2, ..., w_k]
        # objective: min()
        self.SVM_table = {}
        self.iteration = 100
        if ensemble:
            self.lambda_list = np.arange(0.1, 2.0, 0.5) # Test acc : 0.926, Train Acc: 0.918; Test Acc: 0.878, Train Acc: 0.976
        else:
            self.lambda_list = [10.0] # Test Acc: 0.924, Train Acc: 0.914

    def train_single_svm(self, indices, trainX, trainY):
        # trainX: (b, f)
        # trainY: (b, 1)
        # print(f"Shape of X: {trainX.shape}, shape of Y: {trainY.shape}")
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
        # print(batch_size, feature_dim)

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

            # print('====== Finish Solving SVM ======')
            # print(prob)
            # for variable in prob.variables():
            #     print("Variable %s: value %s" % (variable.name(), variable.value))

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
    

# if __name__ == '__main__':
#     from utils import *
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=10)
#     data = prepare_synthetic_data()
#     # data = prepare_mnist_data()
#     svms = MyClassifier(K=3)
#     # svms.train_single_svm((1, 2), data['trainX'], data['trainY'])
#     # new_trainX = pca.fit_transform(data['trainX'])
#     # new_testX = pca.transform(data['testX'])
#     # svms.train(data['trainX'], data['trainY'])
#     svms.train(data['trainX'], data['trainY'])
#     # pred = svms.predict(data['trainX'])
#     # print(pred)
#     acc = svms.evaluate(data['testX'], data['testY'])
#     acc2 = svms.evaluate(data['trainX'], data['trainY'])
#     # acc = svms.evaluate(data['testX'], data['testY'])
#     # acc2 = svms.evaluate(data['trainX'], data['trainY'])
#     print(f"Final Test Prediction Acc: {acc}, Train Acc: {acc2}")

    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        self.cluster_centers_ = None
        self.num_features_ = None
        self.num_train_ = None
    
    def train_one_iter(self, trainX):
        if self.cluster_centers_ is None:
            max_coord = np.max(trainX, axis=0)
            min_coord = np.min(trainX, axis=0)
            self.cluster_centers_ = np.random.uniform(low=min_coord, high=max_coord, size=(self.K, self.num_features_))
        # if self.cluster_centers_ is None:
        #     chosen = set()
        #     _id = np.random.randint(0, trainX.shape[0])
        #     chosen.add(_id)
        #     self.cluster_centers_ = [trainX[_id]]
        #     for i in range(self.K-1):
        #         rem_idxs = []
        #         probs = []
        #         for j in range(trainX.shape[0]):
        #             if j in chosen: continue
        #             rem_idxs.append(j)
        #             mn = float('inf')
        #             for centr in self.cluster_centers_:
        #                 dis = np.linalg.norm(trainX[j]-centr, ord=2)
        #                 mn = min(mn, dis)
        #             assert mn != float('inf')
        #             probs.append(mn**2)
        #         _id = np.random.choice(rem_idxs, p=probs/np.sum(probs))
        #         chosen.add(_id)
        #         self.cluster_centers_.append(trainX[_id])

        norm_mat = np.zeros((self.num_train_, self.K))
        for j in range(self.num_train_):
            cur_sample = trainX[j]
            cur_distance = cur_sample - self.cluster_centers_
            cur_norm = np.square(np.linalg.norm(cur_distance, axis=1))
            norm_mat[j] = cur_norm
        big_M_scale = np.max(norm_mat, axis=0)
        # big_M_scale = np.ones(self.K) * 1000
        radius_vector = cp.Variable(self.K, 'radius')
        binary_labels = cp.Variable((self.num_train_, self.K), 'binary_label')
        objective = cp.Minimize(cp.sum(radius_vector))
        constraints = [binary_labels >= 0, binary_labels <= 1]
        
        for k in range(self.K):
            cur_norm_vector = norm_mat[:, k].flatten() 
            cur_radius, cur_M_scale = radius_vector[k], big_M_scale[k]
            cur_binary_labels = binary_labels[:, k]
            constraints.append(( cur_norm_vector + cur_M_scale * cur_binary_labels <= (cur_radius + cur_M_scale)))
        
        constraints.extend([cp.sum(binary_labels, axis=1) == 1, radius_vector >= 0])
        prob = cp.Problem(
            objective,
            constraints
        )
        prob.solve()
        # print(prob)
        # for variable in prob.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))
        # print(f"Problem status: {prob.status}")
        return radius_vector.value, binary_labels.value
        


    def train(self, trainX, iteration=10):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''
        print('******* K =', self.K)
        self.num_train_, self.num_features_ = trainX.shape
        self.labels = np.zeros(self.num_train_, dtype=int)
        for epoch in range(iteration):
            radius, binary_labels = self.train_one_iter(trainX)
            # print(f"binary labels: {binary_labels[:10, :]}")
            self.labels = np.argmax(binary_labels, axis=1)
            for k in range(self.K):
                class_k_indices = np.where(self.labels == k)[0]
                if len(class_k_indices >= 0):
                    self.cluster_centers_[k] = np.mean(trainX[class_k_indices, :], axis=0)
            # print(f"labels are {self.labels[:10]}, cluster centers are {self.cluster_centers_}")
            print(f'====== Finish Iteration {epoch} of K-means ======')
        # kmeans = KMeans(n_clusters=self.K, max_iter=300).fit(trainX)
        # self.cluster_centers_, self.labels = kmeans.cluster_centers_, kmeans.labels_
        # Update and return the cluster labels of the training data (trainX)
        return self.labels
    
    
    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        pred_labels = None
        num_test = testX.shape[0]
        pred_labels = np.zeros(num_test, dtype=int)
        for t in range(num_test):
            test_sample = testX[t]
            pred_labels[t] = np.argmin(np.linalg.norm(test_sample - self.cluster_centers_, axis=1))
        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self,trainY):
        print(self.labels[:10])
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        # print(trainY[:10])
        # print(aligned_labels[:10])
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            if len(true_labels[index==1]) > 0:
                num = np.bincount(true_labels[index==1]).argmax()
                label_reference[i] = num
        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            if cluster_labels[i] in reference:
                aligned_lables[i] = reference[cluster_labels[i]]
            else:
                aligned_lables[i] = -1
        return aligned_lables

# if __name__ == '__main__':
#     from utils import *

#     # sync_data = prepare_synthetic_data()
#     # kmeans = MyClustering(K=5)
#     # kmeans.train(sync_data['trainX'], iteration=100)
#     # nmi = kmeans.evaluate_clustering(np.array(sync_data['trainY'], dtype=int))
#     # acc = kmeans.evaluate_classification(
#     #     np.array(sync_data['trainY'], dtype=int),
#     #     sync_data['testX'],
#     #     np.array(sync_data['testY'], dtype=int)
#     # )



#     from sklearn.manifold import TSNE
#     tsne = TSNE(n_components=2, perplexity=3, verbose=1)
#     mnist_data = prepare_mnist_data()
#     num_train = len(mnist_data['trainX'])
#     num_test = len(mnist_data['testX'])
#     print(num_train, num_test)
#     allX = np.concatenate([mnist_data['trainX']/255.0, mnist_data['testX']/255.0], axis=0)
#     allX_embedded = tsne.fit_transform(allX)
#     new_trainX = allX_embedded[:num_train]
#     new_testX = allX_embedded[num_train:]

#     kmeans = MyClustering(K=32)
#     kmeans.train(new_trainX, iteration=100)
#     nmi = kmeans.evaluate_clustering(np.array(mnist_data['trainY'], dtype=int))
#     acc = kmeans.evaluate_classification(
#         np.array(mnist_data['trainY'], dtype=int),
#         new_testX,
#         np.array(mnist_data['testY'], dtype=int)
#     )
#     print(f"Final Train NMI: {nmi}, Test ACC: {acc}")

##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio, algo = 'rand', K: int = None):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm
        self.algo = algo
        if self.algo != 'rand':
            self.K = K
        
    def select(self, trainX, debug = False):
        ''' Task 3-2'''
        data_to_label = None
        N = trainX.shape[0]
        M = int(N * self.ratio)
        if self.algo == 'rand':
            return random.sample(range(N), int(N*self.ratio)), None
        elif self.algo == 'dis':
            return self.distance_selection(trainX, M, debug)
        elif self.algo == 'pseudo':
            return self.pseudo_selection(trainX, M, debug)
        else:
            raise NotImplementedError()
    
    def distance_selection(self, trainX: np.ndarray, M: int, debug = False):
        K = self.K
        # kmeans = MyClustering(K)
        # kmeans.train(trainX, 50)
        # centroids, labels = kmeans.cluster_centers_, kmeans.labels
        kmeans = KMeans(n_clusters=K, max_iter=300).fit(trainX)
        centroids, labels = kmeans.cluster_centers_, kmeans.labels_
        dists = [[] for i in range(K)]
        idxs = [[] for i in range(K)]
        lens = [M // K for i in range(K-1)]
        lens.append(M - np.sum(lens))
        for i in range(trainX.shape[0]):
            centr = centroids[labels[i]]
            dists[labels[i]].append(np.linalg.norm(trainX[i] - centr, ord=2))
            idxs[labels[i]].append(i)
        ret = []
        dists = [np.array(dist) for dist in dists]
        for i in range(K):
            sorted_idxs = np.argsort(-dists[i])
            ret += np.array(idxs[i], dtype=np.int32)[sorted_idxs[:lens[i]]].tolist()
        if debug:
            return ret, labels
        return ret
    
    def pseudo_selection(self, trainX: np.ndarray, M: int, debug=True):
        K = self.K
        assert K == 3
        kmeans = KMeans(n_clusters=K, max_iter=300).fit(trainX)
        centroids, labels = kmeans.cluster_centers_, kmeans.labels_
        # kmeans = MyClustering(K)
        # kmeans.train(trainX, 50)
        # centroids, labels = kmeans.cluster_centers_, kmeans.labels
        # print(labels)
        model = MyClassifier(K)
        model.train(trainX, np.array(labels))
        
        vis = set()
        ret = []
        rem = M
        N_svms = len(model.SVM_table.keys())
        cls_to_idxs = dict()
        for x_index in range(trainX.shape[0]):
            x = trainX[x_index, :].reshape(1, -1)
            vote = {}
            for index in model.indices:
                vote[index] = 0
            for indices in model.SVM_table.keys():
                i, j, _ = indices
                svm = model.SVM_table[indices]
                logits = x @ svm[0] + svm[1]
                y_hat = logits > 0
                y_hat = logits.astype(np.int8)
                i_vote = y_hat
                j_vote = 1 - i_vote
                vote[i] += i_vote
                vote[j] += j_vote
            # final_decision = np.argmax(vote)
            vote_list = [(index, vote[index]) for index in model.indices]
            vote_list.sort(key=lambda x : x[1], reverse=True)
            cla = vote_list[0][0]
            if cla not in cls_to_idxs:
                cls_to_idxs[cla] = [x_index]
            else:
                cls_to_idxs[cla].append(x_index)

        for svm_id, indices in enumerate(model.SVM_table.keys()):
            i, j, _ = indices
            svm = model.SVM_table[indices]
            neg_dis = []; pos_dis = []
            pos_idxs_whole = cls_to_idxs[i]
            neg_idxs_whole = cls_to_idxs[j]
            pos_idxs = []; neg_idxs = []
            for idx in pos_idxs_whole:
                if idx in vis: continue
                x = trainX[idx, :].reshape(1, -1)    
                pos_dis.append((x @ svm[0] + svm[1]).squeeze())
                pos_idxs.append(idx)
            for idx in neg_idxs_whole:
                if idx in vis: continue
                x = trainX[idx, :].reshape(1, -1)    
                neg_dis.append((x @ svm[0] + svm[1]).squeeze())
                neg_idxs.append(idx)
            m = rem // (N_svms - svm_id)
            m_pos = min(m//2, len(pos_idxs)); m_neg = m - m_pos
            pos_dis = np.array(pos_dis); neg_dis = np.array(neg_dis)
            pos_idxs = np.array(pos_idxs, dtype=np.int32); neg_idxs = np.array(neg_idxs, dtype=np.int32)
            pos_sort_idx = np.argsort(pos_dis)
            neg_sort_idx = np.argsort(-neg_dis)
            ret += pos_idxs[pos_sort_idx[:m_pos]].tolist()
            ret += neg_idxs[neg_sort_idx[:m_neg]].tolist()
            vis = set(ret)
        vis = set()
        for _id in ret:
            assert _id >= 0 and _id < trainX.shape[0]
            assert _id not in vis
            vis.add(_id)
        if debug:
            return ret, labels
        return ret
            
    
if __name__ == '__main__':
    from utils import *
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    task = 2
    
    if task is 2:
        K = 3
        tsne = TSNE(n_components=2, perplexity=3, verbose=1)
        mnist_data = prepare_mnist_data()
        # mnist_data = prepare_synthetic_data()
        num_train = len(mnist_data['trainX'])
        num_test = len(mnist_data['testX'])
        print(num_train, num_test)
        allX = np.concatenate([mnist_data['trainX']/255.0, mnist_data['testX']/255.0], axis=0)
        # allX = np.concatenate([mnist_data['trainX'], mnist_data['testX']], axis=0)
        allX_embedded = tsne.fit_transform(allX)
        # allX_embedded = allX
        new_trainX = allX_embedded[:num_train]
        new_testX = allX_embedded[num_train:]

        kmeans = MyClustering(K=K)
        kmeans.train(new_trainX, iteration=100)
        nmi = kmeans.evaluate_clustering(np.array(mnist_data['trainY'], dtype=int))
        acc = kmeans.evaluate_classification(
            np.array(mnist_data['trainY'], dtype=int),
            new_testX,
            np.array(mnist_data['testY'], dtype=int)
        )
        print(f"Final Train NMI: {nmi}, Test ACC: {acc}")
    elif task is 1:
        # pca = PCA(n_components=10)
        # data = prepare_synthetic_data()
        data = prepare_mnist_data()
        svms = MyClassifier(K=3, ensemble=True)
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
    else:
        algo = 'pseudo'
        K = 3
        # data = prepare_synthetic_data()
        data = prepare_mnist_data()
        selectors = MyLabelSelection(0.5, algo=algo, K=K)
        idxs, cluster_labels = selectors.select(data['trainX'], True)
        model = MyClassifier(K=3, ensemble=True)
        # print(data['trainY'][idxs])
        model.train(data['trainX'][idxs], data['trainY'][idxs])
        y_hat = model.predict(data['trainX'])
        # print(y_hat)
        plt.scatter(data['trainX'][:, 0], data['trainX'][:, 1], c=y_hat)
        plt.colorbar()
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.savefig('predict.png'); plt.close()
        acc = model.evaluate(data['testX'], data['testY'])
        acc2 = model.evaluate(data['trainX'], data['trainY'])
        print(f"Final Test Prediction Acc: {acc}, Train Acc: {acc2}")
        plt.close()
        
        plt.scatter(data['trainX'][:, 0], data['trainX'][:, 1], c=data['trainY'])
        plt.colorbar()
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.savefig('1.png')
        plt.close()
        plt.scatter(data['trainX'][idxs, 0], data['trainX'][idxs, 1], c=data['trainY'][idxs])
        plt.colorbar()
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.savefig('2.png')
        plt.close()
        # plt.scatter(data['trainX'][:, 0], data['trainX'][:, 1], c=cluster_labels)
        # plt.colorbar()
        # plt.savefig('cluster.png')
        # plt.close()
    
import numpy as np
from .strategy import Strategy
from torch.utils.data import Subset

import pdb
from functools import reduce
from collections import Counter 
from numpy import asarray
from numpy import savetxt
import time
import pickle
class Beyondrc(Strategy):
    def __init__(self, dataset_tr, idxs_lb, train_fun, eval_fun, args):
        super(Beyondrc, self).__init__(dataset_tr, idxs_lb, train_fun, eval_fun, args)

    def query(self, n, model, tokenizer):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        pred_prb = self.predict_prob(model, tokenizer, Subset(self.dataset_tr, idxs_unlabeled))
        probs_sorted, idxs = pred_prb.sort(descending=True)
        probs = probs_sorted[:, 0] - probs_sorted[:,1]#.max(1)[0]
        max_prob = probs_sorted.max(1)[0].numpy()
        probs = probs.numpy()
        with open('newspredprobs.pickle','wb') as f:
            pickle.dump([pred_prb, probs], f)
#         savetxt('probsnew.csv',probs)
        query_learnrate = self.query_learnrate
        thresh = self.thresh/(query_learnrate**self.T)
        ind_probs = np.where(probs <= thresh)[0]#0.9
        num_round = self.num_round
        print('T {} orgthre{} thresh {} learnrate {} weak samples {} round {} '.format(self.T, self.thresh, thresh, query_learnrate, len(ind_probs), num_round), flush=True)
        
        if len(ind_probs) >= n: 
            probs = probs[ind_probs]
        else:
            num_round = num_round + 5
        if True:    
            num_samples = len(probs)
            b_t = min(probs)
            ind_bt = list(probs).index(b_t)
            mu = num_samples
            T = self.T+1
            regret = np.sqrt(T)
            K = num_samples
            probability = np.zeros((num_samples, ))
            delta = 1e-3
            gamma = np.sqrt(K*T/(regret + np.log(2* np.reciprocal(1.0*delta))))
            
            for i in range(len(probs)):
                if i != ind_bt:
                    p_ta = 1/(mu + gamma*( probs[i]-b_t))
                    probability[i] = (p_ta)
            probability[ind_bt] = 1 - np.sum(probability)
            print("max prob {}".format(probability[ind_bt]), flush=True)
            random_ind = np.random.choice(np.arange(num_samples), n, p=list(probability),replace=False)
            tic = time.time()
            for r in range(num_round):
                random_inds = np.random.choice(np.arange(num_samples), n, p=list(probability),replace=False)
                random_ind = np.concatenate((random_ind, random_inds))
            toc = time.time()
            unique, counts = np.unique(random_ind, return_counts=True)
            
            unique = unique.astype(int)
            prob_counts = counts / sum(counts)
    
            counts_ind = np.argsort(-counts)
            cindex = unique[counts_ind][:n]
            new_counts = counts[counts_ind][:n]
            for i in range(len(cindex)): 
                print("gap {} prob_0 {} ind {} count {} ".format( probs[i], max_prob[i], cindex[i], new_counts[i]), flush=True)
            acc_sel = 1
            if ind_bt in cindex:
                print('till now {} / {} best been selected '.format(acc_sel, self.T + 1), flush=True)
            
        if True:#len(ind_probs) >= n: 
            real_index = ind_probs[cindex]
        #else:
        #    real_index = cindex
        realkey = [idxs_unlabeled[ind] for ind in real_index]
        
#         dup_realindex = dict(zip(realkey, new_counts))
        dup_realindex = dict(zip(realkey, np.subtract(new_counts, 1)))
        result_ind = []
        if sum(dup_realindex.values()) > 0:
            result_ind = np.concatenate( [np.repeat(key, dup_realindex[key]) for key in dup_realindex.keys()  if dup_realindex[key]>0] )
# ConcateData        
#         result_ind = []
#         while sum(dup_realindex.values())>0:
#             tmp_ind = []
#             for indx in dup_realindex.keys():
#                 if dup_realindex[indx] > 0:
#                     tmp_ind.append(indx)
#                     dup_realindex[indx] = dup_realindex[indx] -1
#             result_ind.append(tmp_ind)
        
        return idxs_unlabeled[real_index], result_ind

# V1
#         ind_probs = np.where(probs < thresh)[0]#0.9
#         print('T {} threshold {} weak samples {} round {} '.format(self.T, thresh, len(ind_probs), num_round), flush=True)        
        
#         if len(ind_probs) >= n: 
#             probs = probs[ind_probs]
        
#         if True:    
#             num_samples = len(probs)
#             b_t = min(probs)
#             ind_bt = list(probs).index(b_t)
#             mu = num_samples
#             T = self.T+1
#             regret = np.sqrt(T)
#             K = num_samples
#             probability = np.zeros((num_samples, ))
#             delta = 1e-3
#             gamma = np.sqrt(K*T/(regret + np.log(2* np.reciprocal(1.0*delta))))
            
#             for i in range(len(probs)):
#                 if i != ind_bt:
#                     p_ta = 1/(mu + gamma*( probs[i]-b_t))
#                     probability[i] = (p_ta)
#                     #print('i:{} prob:{} select prob: {}'.format(i, probs[i], p_ta), flush=True)
#             probability[ind_bt] = 1 - np.sum(probability)
#             random_ind = np.random.choice(np.arange(num_samples), n, p=list(probability))
#             tic = time.time()
#             for r in range(0):
#                 random_inds = np.random.choice(np.arange(num_samples), n, p=list(probability))
#                 random_ind = np.concatenate((random_ind, random_inds))
#             toc =time.time()
#             print('10 random time {:.2f} s'.format(toc - tic))
#             unique, counts = np.unique(random_ind, return_counts=True) #59000, 34289
#             unique = unique.astype(int)
# # 	    	print("unique:{} counts:{}".format(len(unique), len(counts)), flush=True)
#             counts = counts / sum(counts)
# # 	    	savetxt('./unique.csv', unique, delimiter=',')
#             savetxt('./counts.csv', counts, delimiter=',')
# #data =     loadtxt('data.csv', delimiter=',')
#             counts_ind = np.argsort(-counts)
#             index = unique[counts_ind][:n]#
# #             index = np.random.choice(unique, n, p = counts)
#             sum_prob = []
#             for tmp_index in index: 
#                 sum_prob.append(probs[tmp_index])
# #                 print("sampled gap {} prob:{} ".format( probs[tmp_index], probability[tmp_index]), flush=True)
            
#             print('T:{} w_gap: {:.6f} s_w_gap: {} s_prob: {:.6f} m_gap:{:.6f} s_m_gap {:.6f} mu: {} gamma: {}'.format(T, b_t, np.min(sum_prob), probability[ind_bt], np.mean(probs), np.mean(sum_prob), mu, gamma), flush=True)
#             acc_sel = 1
#             if ind_bt in index:
#                 print('till now {} / {} best been selected '.format(acc_sel, self.T + 1), flush=True)
            
#         if len(ind_probs) >= n: 
#             real_index = ind_probs[index]
#         else:
#             real_index = index
            
#         return idxs_unlabeled[real_index]

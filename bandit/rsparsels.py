import numpy as np
import itertools
import logging
import time
# from RAGE
class bailvscore(object):
    def __init__(self, X, theta_star, factor, delta, sparsity, Z = None):
        
        self.X = X
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(self.Z@theta_star)
        self.delta = delta
        self.factor = factor
        self.omega = range(self.d)
        self.sparsity = sparsity
        
        
    def algorithm(self, seed,  var=True, binary=False):
        
        self.var = var
        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
        self.omega = range(self.d)  
        
        num_iter = 0
        while len(self.active_arms) > 1:
            self.delta_t = self.delta/(self.phase_index**2)      
            
            self.build_Y()
# *******************         Experimental Design
            if True:
                design, rho = self.lev_score()
                support = np.sum((design > 0).astype(int))
                n_min = 2*self.factor*support
                eps = 1/self.factor
                num_samples = max(np.ceil(8*(2**(self.phase_index-1))**2*rho*(1+eps)*np.log(2*self.K_Z**2/self.delta_t)), n_min).astype(int)
                allocation = self.rounding(design, num_samples) # dimension K*1

            pulls = np.array([num*self.X[i] for i, num in enumerate(allocation) if num>0])
            if not binary:
                rewards = pulls@self.theta_star + np.random.randn(pulls.shape[0], 1)
            else:
                rewards = np.random.binomial(1, pulls@self.theta_star, (pulls.shape[0], 1))
            
            ainv = np.linalg.pinv(pulls@pulls.T)
            theta = pulls.T@ainv@rewards
            self.omega = (-abs(theta.flatten())).argsort()[:self.sparsity] # top #sparsity entries

            self.A_inv = (pulls[:,self.omega]).T@ainv@ainv@(pulls[:,self.omega])
            self.diff = np.setdiff1d(np.arange(self.d), self.omega)
            theta[self.diff] = 0
            if np.linalg.norm(theta) != 0:
                logging.info('normalize')
                self.theta_hat = np.linalg.norm(self.theta_star)*theta/np.linalg.norm(theta)
            else:
                self.theta_hat = np.linalg.norm(self.theta_star)*theta
            self.drop_arms()
            logging.info('********* active arm  %d' % len(self.active_arms))
        
            self.phase_index += 1
            self.arm_counts += allocation

            self.N += num_samples
            logging.info('\n\n')
            logging.info('finished phase %s' % str(self.phase_index-1))
            logging.info('design %s' % str(design))
            logging.debug('allocation %s' % str(allocation))
            logging.debug('arm counts %s' % str(self.arm_counts))
            logging.info('round sample count %s' % str(num_samples))
            logging.info('total sample count %s' % str(self.N))
            logging.info('active arms %s' % str(self.active_arms)) 
            logging.info('rho %s' % str(rho))      
            logging.info('\n\n')
            
#             print(self.active_arms)
        
        del self.Yhat
        del self.idxs
        del self.X
        del self.Z
        self.success = (self.opt_arm in self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))
            
    
    def build_Y(self):
        
        k = len(self.active_arms)
        idxs = np.zeros((k*k,2))
        Zhat = self.Z[self.active_arms]
        Y = np.zeros((k*k, self.d))
        rangeidx = np.array(list(range(k)))
        
        for i in range(k):
            idxs[k*i:k*(i+1),0] = rangeidx
            idxs[k*i:k*(i+1),1] = i
            Y[k*i:k*(i+1),:] = Zhat - Zhat[i,:] 
        
        self.Yhat = Y
        self.idxs = idxs
        
    
    def lv_score(self):#experimental design
        
        design = np.ones(self.K)
        design /= design.sum()  
        
        max_iter = 5000
        
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X.T@np.diag(design)@self.X)    
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T
            
            newY = (self.Yhat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))
                        
            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X@A_inv@y)*(self.X@A_inv@y)).flatten()
            g_idx = np.argmax(g)
                        
            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma
                
            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
                        
            design += design_update
            
            if count % 100 == 0:
                logging.debug('design status %s, %s, %s, %s' % (self.seed, count, relative, np.max(rho)))
                            
            if relative < 0.01:
                 break
                        
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        return design, np.max(rho)
    
                
    def rounding(self, design, num_samples):
        
        num_support = (design > 0).sum()
        support_idx = np.where(design>0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - .5*num_support)*support)

        while n_round.sum()-num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round/support)
                n_round[idx] += 1
            else:
                idx = np.argmax((n_round-1)/support)
                n_round[idx] -= 1

        allocation = np.zeros(len(design))
        allocation[support_idx] = n_round
            
        return allocation.astype(int)
      
        
    def drop_arms(self):
        if not self.var:
            print("drop_arms False 1")
            active_arms = self.active_arms.copy()
            removes = set()
            scores = self.Yhat@self.theta_hat
            # gap = 2**(-(self.phase_index+2))
            gap = 2**(-(self.phase_index))

            for t,s in enumerate(scores):
                if gap <= s[0]:
                    arm_idx = int(self.idxs[t][1])
                    removes.add(self.active_arms[arm_idx])

            for r in removes:
                self.active_arms.remove(r)
            
        else:
#             print("drop_arms True 2")
            active_arms = self.active_arms.copy()
            self.theta_hat = self.theta_hat[self.omega]
            for arm_idx in active_arms:
                arm = self.Z[arm_idx, :, None]
                for arm_idx_prime in active_arms:
                    if arm_idx == arm_idx_prime:
                        continue
                    arm_prime = self.Z[arm_idx_prime, :, None]
                    y = arm_prime - arm
                    y = y[self.omega]
#                     if (y.T@self.A_inv@y < 0):
#                         print("A_inv", self.A_inv)
#                         print("y:", y)
#                         print("error!!!!")
#                         break
                    if y.T@self.theta_hat >= np.sqrt(2*y.T@self.A_inv@y*np.log(2*self.K**2/self.delta_t)):
#                         print("minus", y.T@self.theta_hat-np.sqrt(2*y.T@self.A_inv@y*np.log(2*self.K**2/self.delta_t)))
#                         print(arm_idx)
                        self.active_arms.remove(arm_idx)
                        break
         
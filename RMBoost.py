#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from numpy import inf
import numpy as np
from sklearn import tree
import mosek
import random
import scipy

from scipy.io import savemat

import time

import scipy.io as sio
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def predict_boost(model, X):
    T = len(model) - 1
   
    for i in range(T):
        M2_test = model[i].predict(X)
        M2_test_expand = np.expand_dims(M2_test, axis=1)
        if i == 0:
            M_test = M2_test_expand
        else:
         
            M_test = np.concatenate((M_test, M2_test_expand), axis=1)
    y = np.sign(M_test@model[-1])
    return y

def expectation_estimate(i, tau, lmb, M_train, weights, X, y, y_tilde, X_val, y_val, M_val):
    
    n_train = len(y)
    
    random.seed(i)
    idx = np.random.choice(n_train, n_train, replace=True, p=weights)
    
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
    clf = clf.fit(X[idx, :],y_tilde[idx])
    
    M2 = clf.predict(X)
    M2_expand = np.expand_dims(M2, axis=1)
    if i == 0:
        M_train = M2_expand
    else:
        M_train = np.concatenate((M_train, M2_expand), axis=1)

    mm=np.mean(y*M2);
    
    if i == 0:
         tau = np.array([mm])
    else:
        tau = np.concatenate((tau,  np.array([mm])), axis=0)
    
    if len(y_val)>0:
        n_val = len(y_val)
        pred_val = clf.predict(X_val)
        pred_val_expand = np.expand_dims(pred_val, axis=1)
        if i == 0:
            M_val = pred_val_expand
        else:
            M_val = np.concatenate((M_val, pred_val_expand), axis=1)
        tau_val = ((1/n_val)*(np.expand_dims(y_val, axis=1).T@M_val))
        lmb = np.abs(tau - tau_val)
        lmb = np.squeeze(lmb.T)
    else:
        if i == 0:
            lmb = np.array([1/np.sqrt(n_train)])
        else:
            lmb = np.concatenate((lmb, np.array([1/np.sqrt(n_train)])), axis=0)
        M_val = []
        
    return tau, lmb, M_train, M_val, clf

def solver_mosek(i, n_train, c, M):
    with mosek.Env() as env:
        # Create a task
        with env.Task(0, 0) as task:
    
            bkc = [mosek.boundkey.up]*(2*n_train+2*(i+1))
            # Bound values for constraints
            blc = [-inf]*(2*n_train+2*(i+1))
            buc = list(np.squeeze(np.concatenate((0.5*np.ones((2*n_train,1)), np.zeros((2*(i+1),1))), axis=0)))
            # Bound keys for variables
            bkx = [mosek.boundkey.lo]*(2*(i+1))
            # # # Bound values for variables
            blx = [0]*(2*(i+1))
            bux = [inf]*(2*(i+1))
            # Objective coefficients
            csub =  list(range(0, 2*(i+1)))
            cval = list(np.squeeze(c))
            # We input the A matrix column-wise
            # asub contains row indexes
            l = list(range(0, 2*n_train+2*(i+1)))
            asub = l*(2*(i+1))
            # acof contains coefficients
            MT = M.T
            acof = list(np.squeeze(np.reshape(MT, (1, MT.shape[0]*MT.shape[1]))))
            # # aptrb and aptre contains the offsets into asub and acof where
            # # columns start and end respectively
            aptrb = list(range(0, len(asub), 2*n_train+2*(i+1)))
            aptre = list(range(2*n_train+2*(i+1), len(asub)+1, 2*n_train+2*(i+1)))
        
            numvar = len(c)
            numcon = len(bkc)
            
            # Append the constraints
            task.appendcons(numcon)
        
            # Append the variables.
            task.appendvars(numvar)
        
            # Input A non-zeros by columns
            for j in range(numvar):
                # Input objective
                task.putcj(j, cval[j])
                
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                
                ptrb, ptre = aptrb[j], aptre[j]
                task.putacol(j,
                              asub[ptrb:ptre],
                              acof[ptrb:ptre])
            
            for j in range(numcon):
                task.putconbound(j, bkc[j], blc[j], buc[j])
            
            
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
        
            # Optimize the task
            task.optimize()
            
            # task.solutionsummary(mosek.streamtype.msg)
            
            solsta = task.getsolsta(mosek.soltype.bas)
            R = task.getprimalobj(mosek.soltype.bas)+1/2
            
            # Output a solution
            x_mosek = task.getxx(mosek.soltype.itr)
            # a0 = x_mosek[0:(i+1)]
            # a1 = x_mosek[(i+1):(2*i+2)]
            # mu_mosek=[a - b for a, b in zip(a0, a1)]
            t = task.getsolution(mosek.soltype.bas)
            dual_sol = t[7]
    
    return x_mosek, dual_sol, R

def solver_linprog(i, n_train, c, M):
    
    buc = list(np.squeeze(np.concatenate((0.5*np.ones((2*n_train,1)), np.zeros((2*(i+1),1))), axis=0)))

    cval = list(np.squeeze(c))

    sol = scipy.optimize.linprog(cval, A_ub=M, b_ub=buc, A_eq=None, b_eq=None, bounds=None, method='highs', callback=None, options=None, x0=None, integrality=None)

    R = sol.fun+1/2
    x = sol.x
    
    dual_sol = sol.ineqlin.marginals
    
    return x, dual_sol, R
        
def iboost(i, tau, lmb, M_train, weights, X, y, y_tilde, solver, X_val, y_val, M_val):
    
    tau, lmb, M_train, M_val, clf = expectation_estimate(i, tau, lmb, M_train, weights, X, y, y_tilde, X_val, y_val, M_val)

    M_c1 = np.concatenate((M_train, -M_train), axis = 0)
    M_c1 = np.concatenate((M_c1, -np.identity(i+1)), axis = 0)
    M_c1 = np.concatenate((M_c1, np.zeros((i+1, i+1))), axis = 0)
    
    M_c2 = np.concatenate((-M_train, M_train), axis = 0)
    M_c2 = np.concatenate((M_c2, np.zeros((i+1, i+1))), axis = 0)
    M_c2 = np.concatenate((M_c2, -np.identity(i+1)), axis = 0)
    
    n_train = len(y)
    
    M=np.concatenate((M_c1, M_c2), axis=1)
    c = np.concatenate((-tau+lmb, tau+lmb), axis=0)
    
    if solver == "Mosek":

        x, dual_sol, upper = solver_mosek(i, n_train, c, M)
        
    elif solver == "linprog":

        x, dual_sol, upper = solver_linprog(i, n_train, c, M)
        
    alpha=dual_sol[0:n_train]
    beta=dual_sol[n_train+1:2*n_train+1]
    
    y_tilde=np.sign(y/n_train-alpha+beta)
    weights=np.abs(y/n_train-alpha+beta)

    weights = weights/weights.sum()
    
    a0 = x[0:(i+1)]
    a1 = x[(i+1):(2*i+2)]
    mu=[a - b for a, b in zip(a0, a1)]

    return tau, lmb, M_train, y_tilde, mu, upper, M_val, clf
        

def fit(X, y_train, T = 2000, solver = "Mosek", n_samples = 1000):
    
    n_train = len(y_train)
    
    if n_train > n_samples:
        X, X_val, y_train, y_val = train_test_split(X, y_train, train_size=n_samples/n_train)
    else:
        X_val = []
        y_val = []
        
    n_train = len(y_train)
    weights=np.ones((n_train,))/n_train
    tau = []
    lmb = []
    M_train = []
    M_val = []
    y_tilde = y_train
    upper = np.zeros((T,))
    model = np.zeros((T+1,), dtype = object)
  

        
    for i in range(0, T):
        
        tau, lmb, M_train, y_tilde, mu, R, M_val, clf = iboost(i, tau, lmb, M_train, weights, X, y_train, y_tilde, solver, X_val, y_val, M_val)
        
        upper[i] = R
        model[i] = clf
    model[i+1] = mu
            
    return model, upper
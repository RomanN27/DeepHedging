# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:01:19 2021

@author: roman
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from data import *
from functools import partial 

import ipympl
import os 
#%matplotlib inline
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from TemporalConvNet import *

params= pd.read_csv("NSS_params2.csv", sep=";", decimal=",")

def bm(t,delta_t):
    n=int(t/delta_t)
    dW=np.random.normal(0,delta_t,n)
    W= np.cumsum(dW)
    return W

def Svensson(params,tenor, mode="yield", params_type="dec"):
        
        b_0,b_1,b_2,b_3,tau_1,tau_2 = [params[i] for i in range(len(params))]
        S_1=b_0+b_1*(1- np.exp(-tenor/tau_1))/(tenor/tau_1)
        S_2=b_2*((1-np.exp(-tenor/tau_1))/(tenor/tau_1) - np.exp(-tenor/tau_1))
        S_3=b_3*((1-np.exp(-tenor/tau_2))/(tenor/tau_2)-np.exp(-tenor/tau_2))
        S=S_1+S_2+S_3 
        if(params_type=="procent"): 
           S=S/100
        if mode == "yield":
            return S
        if mode == "zero":
            P=np.exp(-S*tenor)
            return P
        if mode== "discont":
            return (1+S)**-tenor

class Ho_Lee():
    def __init__(self, Zero_curve,sigma):
        self.P=Zero_curve
        self.sigma=sigma
        self.forward_rates_help()
        self.Variance_int()
    def short_rate_sampler(self, T, dt):
        time=np.round( np.arange(int(T/dt))*dt+dt ,2)
        #T/dt has to be in sigma[:, 0]
        index=np.where(self.sigma[:,0]==float(T))
        time_diff=np.diff(self.sigma[:,0], prepend=0)
        #t of sigma has to be multiple of dt
        time_counts=np.round(time_diff/dt,2)
        sigma_expanse=np.repeat(self.sigma[:,1] , time_counts.astype(int))
        
        forward_rates=np.gradient(self.f(time),dt)
        
        V=self.V(time)
        V_1=np.gradient(V,dt)
        V_2=np.gradient(V_1,dt)
        theta=np.gradient(forward_rates,dt)+0.5*(V_2)
        r=np.cumsum(theta* dt + sigma_expanse*np.random.normal(0, dt,size=sigma_expanse.shape)).reshape(-1,1)
        #plt.plot(r)
        time=time.reshape(-1,1)
        r=np.concatenate((time,r), axis=1)
        
        return r
    def forward_rates_help(self):
        f= lambda t : np.log(self.P(t))
        self.f=f
        
    def Variance_int(self):
        def V(T):
            # if(T[-1] > np.max(self.sigma[:,0])):
            #     sigma= np.concatenate((self.sigma, np.array([[T[-1], self.sigma[-1,1]]])))
            # elif(T[0]<np.min(self.sigma[:,0])):
            #     sigma= np.concatenate((np.array([[T[0], self.sigma[0,1]]]),self.sigma))
            # else:
            #     sigma=self.sigma
            
            
            t_max= np.array([np.where(sigma[:,0]>=tt)[0][0] for tt in  T])
            t_max[np.where(t_max==0)]=1
            delta_t = T-sigma[:,0][t_max-1] 
            time_diffs= np.diff(sigma[:,0],prepend=0)
            V=np.array([np.sum(sigma[:t,1]*time_diffs[:t]) + delta_t[i]*sigma[t,1] for i,t in enumerate(t_max)])
            return V
        self.V=V

    def Zero_Bond(self,r,T):
        t=r[:,0]
        T=T.reshape(-1,1)
        B=T
        A=np.log(self.P(T)/self.P(t)) + B*self.f(t)-0.5*B**2 * self.V(t)
        P_t= np.exp(A-B*r[:,1])
        return P_t

def derivative(f,eps=10**-5):
    def f_1(x):
        return (f(x)-f(x-eps))/eps
    return f_1 



param=params.iloc[-1,1:]


P=partial(Svensson, param, params_type="procent", mode="zero")


def sigma_sample(dt,T):
    time= np.round((np.arange(int(T/dt)))*dt+dt,2)
    sigma= np.abs(np.random.normal(0,0.1, len(time)))
    sigma=np.concatenate((time.reshape(-1,1),sigma.reshape(-1,1)) ,axis=1)
    return sigma


dt=0.1
T=30
sigma=sigma_sample(dt,T)


test=Ho_Lee(P,sigma)
r=test.short_rate_sampler(T, dt)
plt.plot(r[:,1])
maturities=np.array([0.25,0.5,0.75,1,2,3,4,5,7,10,15,20,30,40,50])
P=test.Zero_Bond(r, maturities)


Deep

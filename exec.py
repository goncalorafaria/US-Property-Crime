"""
# Header ------------------------------------------------------------------
"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from core import *
from model import build_propertycrime
from scipy.stats import poisson
import sys
np.set_printoptions(precision=10)

# data load  ----------------------------------------------------------

np.random.seed(123)

df = pd.read_csv("experiment1.csv")

models=build_propertycrime()

# dat treatment ----------------------------------------------------------

T = np.unique(df["Year"]).shape[0]

d15n=df.loc[ df["Year"]== 2015, ["Population"] ].values
d16n=df.loc[ df["Year"]== 2016, ["Population"] ].values

d15y=df.loc[ df["Year"]== 2015, ["Property crime"] ].values
d16y=df.loc[ df["Year"]== 2016, ["Property crime"] ].values

n=np.array([d15n, d16n]).squeeze(-1).T
y=np.array([d15y, d16y]).squeeze(-1).T

df["Regions"] = pd.Categorical(df["Regions"])
df["Year"] = pd.Categorical(df["Year"])

region = df["Regions"].cat.codes.values.reshape(-1)
year = df["Year"].cat.codes.values.reshape(-1)

nregions = np.unique(region).shape[0]
nstates = n.shape[0]

## diagnostics ----------------------------------------------------------

def estimate_pd( log_likelihood ,chosen_log_likelihood_mean):
    expected_log_likelihood_mean = np.mean(
        log_likelihood
    )
    
    pw = - 2 * ( expected_log_likelihood_mean -  chosen_log_likelihood_mean )
    return pw

def estimate_waic(likelihood_samples, pw):

    expected_density_per_datapoint = np.mean(likelihood_samples, axis=(-1,-2) )
    waic_c = - 2 * ( np.sum( np.log( expected_density_per_datapoint ) ) - pw )

    return waic_c

def estimate_dic(pd, chosen_log_likelihood_mean):
    
    dic_c = - 2 * (chosen_log_likelihood_mean - pd)

    return dic_c

def eplot(model, y, n, it = 10000, chains=4,sslice =1000, debug = False):
    samples = sample(model[0], model[2], varnames=model[1], chains=chains, trials=it)
    
    #if debug:
    #    plt.hist([np.array(samples['y_pred'][:,:,-sslice,:].ravel()), 
    #    np.array(y.ravel())],bins=10,histtype='bar',density=True,label=("predicted","true"))

    #    plt.legend(loc="upper right")
    #    plt.show()
    # 
    # 'log.likelihood', 'likelihood',
        
    y_pred = np.mean( samples['y_pred'][:,:,-sslice:,:], axis=(-1,-2) )

    chosen = np.mean( samples['lambda'][:,:,-sslice:,:], axis=(-1,-2) ).ravel()
    
    chosen_log_likelihood_mean = np.sum(poisson.logpmf(y.ravel(),chosen))

    pw = estimate_pd( samples['log.lik'][:,-sslice:,:] ,chosen_log_likelihood_mean)
    waic = estimate_waic(samples['likelihood'][:,:,-sslice:,:], pw)
    dic = estimate_dic(pw, chosen_log_likelihood_mean)
    mse = np.mean((y.ravel()-y_pred.ravel())**2) 
    likelihood = np.mean(samples['log.lik'][:,-sslice:,:].ravel())

    print("pd :" + str(pw))
    print("dic :" + str(dic))
    print("waic :" + str(waic))
    print("mse :" + str(mse))
    print("likelihood :" + str(likelihood))
    
    print("######################")
    if debug:
        analyse_fit(model[1],samples)
    
        plot_predictive(
            n.ravel(),
            y.ravel(), 
            y_pred.ravel())
        
    plt.show()


for k, model in models.items():
    print(" Model: " + str(k))

    eplot(model, y = y, n=n, it=40000, sslice=5000)

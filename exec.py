"""
# Header ------------------------------------------------------------------
"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from core import *
from model import build
from scipy.stats import poisson
import sys
np.set_printoptions(precision=10)

# data load  ----------------------------------------------------------

np.random.seed(123)

df = pd.read_csv("experiment1.csv")

models=build()

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

def eplot(model, y, n, it = 10000, chains=4,sslice =1000, debug = False):
    samples = sample(model[0], model[2], varnames=model[1], chains=chains, trials=it)
    
    #if debug:
    #    plt.hist([np.array(samples['y_pred'][:,:,-sslice,:].ravel()), 
    #    np.array(y.ravel())],bins=10,histtype='bar',density=True,label=("predicted","true"))

    #    plt.legend(loc="upper right")
    #    plt.show()
        
    y_pred = np.mean( samples['y_pred'][:,:,-sslice:,:], axis=(-1,-2) )
    loglikl = np.mean(
        samples['log.lik'][:,-sslice:,:]
        .ravel()
    )
    _lambda_me = np.mean( samples['lambda'][:,:,-sslice:,:], axis=(-1,-2) )
    _lambda_mi = np.median( samples['lambda'][:,:,-sslice:,:], axis=(-1,-2) )
    
    print("######################")
    print("Log likelihood SMP: " + str(loglikl)) 
    print("######################")
    
    loglikl_mean = np.sum(poisson.logpmf(y.ravel(),_lambda_me.ravel()))
    loglikl_median = np.sum(poisson.logpmf(y.ravel(),_lambda_mi.ravel()))
    
    loglikl_mean_presented = np.sum(poisson.logpmf(y_pred.ravel(),_lambda_me.ravel()))
    loglikl_median_presented = np.sum(poisson.logpmf(y_pred.ravel(),_lambda_mi.ravel()))
    
    y_predmean = n.ravel() * _lambda_me.ravel()
    mse_mean = np.sum(((y.ravel()-y_predmean)/y.ravel())**2) 
    
    y_predmd = n.ravel() *_lambda_mi.ravel()
    mse_md = np.sum(((y.ravel()-y_predmd)/y.ravel())**2) 
    
    aic_mean = 2* model[3] - 2 * loglikl_mean
    aic_md = 2* model[3] - 2 * loglikl_median
    sic_mean = 2* np.log(len(y.ravel()))*model[3] - 2 * loglikl_mean
    sic_md = 2* np.log(len(y.ravel()))*model[3] - 2 * loglikl_median
    
    
    ##
    pd = - 2 * loglikl + 2 * loglikl_mean 
    
    print("pd :" + str(pd))
    print("dic :" + str( -2 * loglikl_mean + 2 * pd))
    ##
    
    print("######################")
    
    print("### mean: ---")
    print("Log likelihood: " + str(loglikl_mean)) 
    print("Log likelihood pred : " + str(loglikl_mean_presented)) 
    print("aic: "+ str(aic_mean))
    print("sic: "+ str(sic_mean))
    print("dic:")
    print("chi-square : " + str(mse_mean))
    
    print("### median: ---")
    print("Log likelihood: " + str(loglikl_median))
    print("Log likelihood pred : " + str(loglikl_median_presented)) 
    print("aic: "+ str(aic_md))
    print("sic: "+ str(sic_md))
    print("chi-square : " + str(mse_md))
    
    print("### sampled: ---")
    print("mse: " + str(np.sum(((y.ravel()-y_pred.ravel())/y.ravel())**2) ) )
    
    print("######################")
    if debug:
        analyse_fit(model[1],samples)
    
    plot_predictive(
        n.ravel(),
        y.ravel(), 
        y_pred.ravel())
    
    eps = np.mean( ( y - y_pred) ** 2 / y ) 
    print(eps)
    
    plt.show()

eplot(models[sys.argv[1]], y = y, n=n, it=int(sys.argv[2]), sslice=4000, debug=True)

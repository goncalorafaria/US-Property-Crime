"""
# Header ------------------------------------------------------------------
"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from core import *
from model import build_propertycrime, build_subtypes, build_full
from scipy.stats import poisson, multinomial
import sys
import pickle

np.set_printoptions(precision=10)

# data load  ----------------------------------------------------------

np.random.seed(123)

df = pd.read_csv("experiment1.csv")

# dat treatment ----------------------------------------------------------

T = np.unique(df["Year"]).shape[0]

d15n=df.loc[ df["Year"]== 2015, ["Population"] ].values
d16n=df.loc[ df["Year"]== 2016, ["Population"] ].values

d15y=df.loc[ df["Year"]== 2015, ["Property crime"] ].values
d16y=df.loc[ df["Year"]== 2016, ["Property crime"] ].values

d15a=df.loc[ df["Year"]== 2015, ["Burglary"] ].values
d16a=df.loc[ df["Year"]== 2016, ["Burglary"] ].values

d15b=df.loc[ df["Year"]== 2015, ["Larceny-theft "] ].values
d16b=df.loc[ df["Year"]== 2016, ["Larceny-theft "] ].values

d15c=df.loc[ df["Year"]== 2015, ["Motor vehicle theft "] ].values
d16c=df.loc[ df["Year"]== 2016, ["Motor vehicle theft "] ].values

n=np.array([d15n, d16n]).squeeze(-1).T
y=np.array([d15y, d16y]).squeeze(-1).T
a=np.array([d15a, d16a]).squeeze(-1).T
b=np.array([d15b, d16b]).squeeze(-1).T
c=np.array([d15c, d16c]).squeeze(-1).T

df["Area"] = pd.Categorical(df["Area"])
df["Regions"] = pd.Categorical(df["Regions"])
df["Year"] = pd.Categorical(df["Year"])

states = df["Area"].cat.codes.values.reshape(-1)
region = df["Regions"].cat.codes.values.reshape(-1)
year = df["Year"].cat.codes.values.reshape(-1)

nregions = np.unique(region).shape[0]
nstates = n.shape[0]

p = np.stack([a,b,c],axis=-1).astype(int)

yn = np.sum(p, axis=-1)

k=3

## diagnostics ----------------------------------------------------------

def estimate_pd( log_likelihood, chosen_log_likelihood_mean):
    expected_log_likelihood_mean = np.mean(
        log_likelihood
    )

    pw = - 2 * ( expected_log_likelihood_mean -  chosen_log_likelihood_mean )
    return pw

def estimate_pw(likelihood_samples, log_likelihood_samples):
    
    expected_density_per_datapoint = np.mean(likelihood_samples, axis=(-1,-2))

    mfit = np.sum( np.log( expected_density_per_datapoint ) )
    
    expected_log_density_per_datapoint = np.mean(
        log_likelihood_samples, axis=(-1,-2))
    
    exlog = np.sum(expected_log_density_per_datapoint)
    
    pw = -2 * (exlog - mfit)
    
    return pw

def estimate_waic(likelihood_samples, pw):

    expected_density_per_datapoint = np.mean(likelihood_samples, axis=(-1,-2) )
    waic_c = - 2 * ( np.sum( np.log( expected_density_per_datapoint ) ) - pw )

    return waic_c

def estimate_dic(pd, chosen_log_likelihood_mean):
    
    dic_c = - 2 * (chosen_log_likelihood_mean - pd)

    return dic_c

def splot(model, y, p, it = 10000, chains=4,sslice =1000, debug = False, k=3):

    samples = sample(model[0], model[2], varnames=model[1], chains=chains, trials=it)

    chosen = np.mean( samples['rate'][:,:,:,-sslice:,:], axis=(-1,-2) ).reshape((-1,k))
    y_pred = np.mean( samples['y_pred'][:,:,:,-sslice:,:], axis=(-1,-2) ).reshape((-1,k))

    chosen = chosen / np.sum(chosen,axis=-1,keepdims=True)

    chosen_log_likelihood_mean = np.sum(
        multinomial.logpmf(p.reshape((-1,k)) ,n = y.ravel(), p = chosen)
    )

    print(chosen_log_likelihood_mean)


    pd = estimate_pd(samples['log.likelihood_sum'][:,-sslice:,:] ,chosen_log_likelihood_mean)
    
    pw = estimate_pw(
        samples['likelihood'][:,:,-sslice:,:], 
        samples['log.likelihood'][:,:,-sslice:,:])

    waic = estimate_waic(samples['likelihood'][:,:,-sslice:,:], pw)
    dic = estimate_dic(pd, chosen_log_likelihood_mean)
    mse = np.mean((p.ravel()-y_pred.ravel())**2) 
    likelihood = np.mean(samples['log.likelihood_sum'][:,-sslice:,:].ravel())

    print("######################")
    print("pd :" + str(pd))
    print("pw :" + str(pw))
    print("dic :" + str(dic))
    print("waic :" + str(waic))
    print("mse :" + str(mse))
    print("likelihood :" + str(likelihood))
    print("chosen likelihood :" + str(chosen_log_likelihood_mean))

    print("######################")

    if debug:
        analyse_fit(model[1],samples)
        plt.show()

    return samples

def eplot(model, y, n, it = 10000, chains=4,sslice =1000, debug = False):
    samples = sample(model[0], model[2], varnames=model[1], chains=chains, trials=it)
        
    y_pred = np.mean( samples['y_pred'][:,:,-sslice:,:], axis=(-1,-2) )

    chosen = np.mean( samples['lambda'][:,:,-sslice:,:], axis=(-1,-2) ).ravel()
    
    chosen_log_likelihood_mean = np.sum(poisson.logpmf(y.ravel(),chosen))

    pd = estimate_pd( samples['log.lik'][:,-sslice:,:] ,chosen_log_likelihood_mean)
    pw = estimate_pw(
        samples['likelihood'][:,:,-sslice:,:], 
        samples['log.likelihood'][:,:,-sslice:,:])

    waic = estimate_waic(samples['likelihood'][:,:,-sslice:,:], pw)
    dic = estimate_dic(pd, chosen_log_likelihood_mean)
    mse = np.mean((y.ravel()-y_pred.ravel())**2) 
    likelihood = np.mean(samples['log.lik'][:,-sslice:,:].ravel())

    
    print("pd :" + str(pd))
    print("pw :" + str(pw))

    print("dic :" + str(dic))
    print("waic :" + str(waic))
    print("mse :" + str(mse))
    print("likelihood :" + str(likelihood))
    print("chosen_likelihood :" + str(chosen_log_likelihood_mean))
    
    print("######################")
    if debug:
        analyse_fit(model[1],samples)
    
        plot_predictive(
            n.ravel(),
            y.ravel(), 
            y_pred.ravel())
        
    plt.show()
    return samples

def fplot(model, yn, n,p, it =10000, chains=4, sslice=1000, debug= False, k=3):

    samples = sample(model[0], model[2], varnames=model[1], chains=chains, trials=it)

    chosen_rate = np.mean( samples['rate'][:,:,:,-sslice:,:], axis=(-1,-2))

    chosen_rate = chosen_rate / np.sum(chosen_rate,axis=-1,keepdims=True)

    chosen_rate_log_likelihood_mean = np.sum(
        multinomial.logpmf(p.reshape((-1,k)) ,n = yn.ravel(), p = chosen_rate.reshape((-1,k)))
    )

    chosen_lambda = np.mean( samples['lambda'][:,:,-sslice:,:], axis=(-1,-2) ).ravel()

    chosen_lambda_log_likelihood_mean = np.sum(poisson.logpmf(yn.ravel(),chosen_lambda))

    total_chosen = chosen_rate_log_likelihood_mean + chosen_lambda_log_likelihood_mean
    
    pd = estimate_pd( samples['log.likelihood_sum'][:,-sslice:,:] ,total_chosen)
    pw = estimate_pw(
        samples['likelihood'][:,:,-sslice:,:], 
        samples['log.likelihood'][:,:,-sslice:,:])

    waic = estimate_waic(samples['likelihood'][:,:,-sslice:,:], pw)
    dic = estimate_dic(pd, total_chosen)

        
    print("pd :" + str(pd))
    print("pw :" + str(pw))

    print("dic :" + str(dic))
    print("waic :" + str(waic))
    print("chosen_likelihood :" + str(total_chosen))
        
    print("######################")



models = build_propertycrime()

for k,v in models.items():
    print("Model")
    print(k)
    eplot(v, y, n, it = 10000, chains=4,sslice =1000, debug = False)

models = build_subtypes()
for k,v in models.items():
    print("Model")
    print(k)
    splot(v, y, p, it = 10000, chains=4,sslice =1000, debug = False, k=3)

models = build_full()
for k,v in models.items():
    print("Model")
    print(k)
    fplot(v, y, n ,p , it =10000, chains=4, sslice=1000, debug= False, k=3)

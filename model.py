"""
# Header ------------------------------------------------------------------
"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from core import *
from scipy.stats import poisson
#np.set_printoptions(precision=10)

def get_ind(n, iso):
    l1 = np.arange(0,n)
    l2 = iso
    return [x for x in l1 if x not in l2], list(iso)

def build_region_proximity_matrix():
    Wr_dict = {
        0 : [8,2,1,7],
        1 : [0,7, 9],
        2 : [0,4,7],
        3 : [5,8,9],
        4 : [2],
        5 : [3],
        6 : [],
        7 : [1,2,0],
        8 : [3,0,9],
        9 : [1,8,3],
    }

    Wr = np.zeros((10,10))

    for i,v in Wr_dict.items():
        for j in v:
            Wr[i,j]=1
            Wr[j,i]=1

    isor = np.where( np.sum(Wr, axis=1) <= 1e-2 )[0]
    nisor, isor = get_ind(10,isor)

    Wr = np.delete(Wr, isor, 0)
    Wr = np.delete(Wr, isor, 1)
        
    Dr = np.diag(np.sum(Wr,axis=1))  

    Ir = np.eye(Wr.shape[0])
    Wrt = np.linalg.inv(Dr)@Wr

    return Wr, isor, nisor, Dr, Ir, Wrt

def build_state_proximity_matrix():
    # W proximity matrix for states -

    ndict = {}
    graph = {}
    rv_dict = {}
    with open("state_neighbors.txt", "r") as f :
        with open("state_names.txt", "r") as fn:
            while True:

                line = f.readline()
                linen= fn.readline()
                if not line:
                    break
                chain = line.split("\n")[0].split(" ")
                name = linen.split("\n")[0]

                head = chain[0]
                tail = chain[1:]
                            
                if head != "US":
                    if head == "UT":
                        tail = tail[:-1]
                    ndict[head] = name
                    rv_dict[name] = head
                    graph[head] = tail

    i=0
    code = {}
    for e in np.sort(list(rv_dict.keys())):
        code[e] = i
        i+=1
        
    Ws = np.zeros( (52,52) )

    for a, adj in graph.items():
        for b in adj:
            codea = code[ndict[a]]
            codeb = code[ndict[b]]
            Ws[codea,codeb] = 1
            Ws[codeb,codea] = 1

    isos = np.where( np.sum(Ws, axis=1) <= 1e-2 )[0]
    nisos, isos = get_ind(52,isos)
    Ws = np.delete(Ws, isos, 0)
    Ws = np.delete(Ws, isos, 1)
        
    Ds = np.diag(np.sum(Ws, axis=1))

    Is = np.eye(Ws.shape[0])
    Wst = np.linalg.inv(Ds)@Ws

    return Ws, isos, nisos, Ds, Is, Wst  

def mcodes(region_prior,state_prior,n , y, yeari, regioni, statei):

    Wr, isor, nisor, Dr, Ir, Wrt = region_prior 
    Ws, isos, nisos, Ds, Is, Wst = state_prior 

    year, T = yeari 
    region, nregions = regioni
    states, nstates = statei 

    models={}
    
    ######

     # In[192]:

    ### PARSAR Model
    modelcode = '''
    model
    {
        # Likelihood

        for( s in 1:nstates ){ theta[s,1] <- 0 }
                
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- neffect + reffect[region[s]+1] + sieffect[s,t] + seffect[s]
                
                theta[s,t+1] <- mu.effect[s,t] 
                
                lambda[s,t] <- theta[s,t+1] * n[s,t]

                y[s,t] ~ dpois(lambda[s,t])
                y_pred[s,t] ~ dpois(lambda[s,t])                     

                plog.lik[s,t] <- logdensity.pois(y[s,t], lambda[s,t])

            }
        }

        log.lik <- sum(plog.lik)

        #### Parameters: ############### 
        # Priors
        #--- Time
        
        t.c ~ dnorm(0,1.0e-3)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 
    
        ##--- regional sas
    
        sar.phi ~ dunif(-0.99, 0.99)
        sar.tau ~ dgamma( 1.0E-3,  1.0E-3)

        sar.mu ~ dmnorm(rep(0, length(nisor)), 1.0E-3 * Ir)
                
        sar ~ dmnorm(sar.mu, sar.tau * Dr %*% (Ir - sar.phi*Wr) )
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- sar[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, sar.tau) }
        
        # --- Epsilon

        ## iid part

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        } 

        ## auto regressive errors
        sas.phi ~ dunif(-0.99, 0.99)
        sas.tau ~ dgamma( 1.0E-3,  1.0E-3)
        w ~ dnorm(0, 1e-2)
        
        sas ~ dmnorm(rep(0, length(nisos)), sas.tau * Ds %*% (Is - sas.phi*Ws))

        ## first element of the series AR not applied

        for (i in 1:length(nisos)) { 
            seffect[nisos[i]+1] <- sas[i] 
        }
        
        for (i in 1:length(isos)) { 
            isoeffect[i] ~ dnorm(0, sas.tau);
            seffect[isos[i]+1] <- isoeffect[i];
        }

    }
    '''

    varnames = ['y_pred','theta','log.lik','t.c','w','sar.phi','sas.phi','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,
    region=region, y=y, n=n, T=T, 
    isos=isos, nisos=nisos, Is=Is, Ds=Ds,Ws=Ws,
    isor=isor, nisor=nisor, Ir=Ir, Dr=Dr,Wr=Wr)
    models["IV"] = (modelcode,varnames,bvars,par)


    ### PARSAR Model
    modelcode = '''
    model
    {
        # Likelihood

        for( s in 1:nstates ){ theta[s,1] <- 0 }
                
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- neffect + sieffect[s,t] + seffect[s]
                
                theta[s,t+1] <- mu.effect[s,t] 
                
                lambda[s,t] <- theta[s,t+1] * n[s,t]

                y[s,t] ~ dpois(lambda[s,t])
                y_pred[s,t] ~ dpois(lambda[s,t])                     

                plog.lik[s,t] <- logdensity.pois(y[s,t], lambda[s,t])

            }
        }

        log.lik <- sum(plog.lik)

        #### Parameters: ############### 
        # Priors
        #--- Time
        
        t.c ~ dnorm(0,1.0e-3)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 

        # --- Epsilon

        ## iid part

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        } 

        ## auto regressive errors
        sas.phi ~ dunif(-0.99, 0.99)
        sas.tau ~ dgamma( 1.0E-3,  1.0E-3)
        w ~ dnorm(0, 1e-2)
        
        sas ~ dmnorm(rep(0, length(nisos)), sas.tau * Ds %*% (Is - sas.phi*Ws))

        ## first element of the series AR not applied

        for (i in 1:length(nisos)) { 
            seffect[nisos[i]+1] <- sas[i] 
        }
        
        for (i in 1:length(isos)) { 
            isoeffect[i] ~ dnorm(0, sas.tau);
            seffect[isos[i]+1] <- isoeffect[i];
        }

    }
    '''

    varnames = ['y_pred','theta','log.lik','w','sas.phi','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,y=y, n=n, T=T, 
    isos=isos, nisos=nisos, Is=Is, Ds=Ds,Ws=Ws)

    models["III"] = (modelcode,varnames,bvars,par)


    ###

    modelcode = '''
    model
    {
        # Likelihood

        for( s in 1:nstates ){ theta[s,1] <- 0 }
                
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- neffect + sieffect[s,t]
                
                theta[s,t+1] <- mu.effect[s,t] 
                
                lambda[s,t] <- theta[s,t+1] * n[s,t]

                y[s,t] ~ dpois(lambda[s,t])
                y_pred[s,t] ~ dpois(lambda[s,t])                     

                plog.lik[s,t] <- logdensity.pois(y[s,t], lambda[s,t])

            }
        }

        log.lik <- sum(plog.lik)

        #### Parameters: ############### 
        # Priors
        #--- Time
        
        w ~ dnorm(0, 1e-2)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 

        # --- Epsilon

        ## iid part

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        } 

        ## auto regressive errors

    }
    '''

    varnames = ['y_pred','theta','log.lik','w','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,y=y, n=n, T=T)
    
    models["I"] = (modelcode,varnames,bvars,par)

    ###

    modelcode = '''
    model
    {
        # Likelihood
                
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- neffect + reffect[region[s]+1] + sieffect[s,t]
                
                theta[s,t] <- mu.effect[s,t] 
                
                lambda[s,t] <- theta[s,t]* n[s,t]

                y[s,t] ~ dpois(lambda[s,t])
                y_pred[s,t] ~ dpois(lambda[s,t])     
                

                plog.lik[s,t] <- logdensity.pois(y[s,t], lambda[s,t])

                #msei_p[s,t] <- sieffect[s,t] ^ 2 
                #msed_p[s,t] <- seffect[s,t] ^ 2
            }
        }
        #msei <- mean(msei_p)
        #msed <- mean(msed_p)

        #mse <- msei + msed

        log.lik <- sum(plog.lik)

        #### Parameters: ############### 
        # Priors
        #--- Time
        
        w ~ dnorm(0,1.0e-3)
        #t.c ~ dnorm(0,1.0e-3)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 
    
        ##--- regional sas
    
        sar.phi ~ dunif(-0.99, 0.99)
        sar.tau ~ dgamma( 1.0E-3,  1.0E-3)

        sar.mu ~ dmnorm(rep(0, length(nisor)), 1.0E-3 * Ir)
                
        sar ~ dmnorm(sar.mu, sar.tau * Dr %*% (Ir - sar.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- sar[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, sar.tau) }
        
        # --- Epsilon

        ## iid part

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        }   
        
        
    }
    '''

    varnames = ['y_pred','theta','log.lik','sar.phi','lambda','sar.mu','sar']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,
    region=region, y=y, n=n, T=T, 
    isor=isor, nisor=nisor, Ir=Ir, Dr=Dr, Wr=Wr)
    models["II"] = (modelcode,varnames,bvars,par)

    ###
    return models


def build(seed = 123):

    np.random.seed(seed)

    
    region_prior = build_region_proximity_matrix()
    state_prior = build_state_proximity_matrix()

    df = pd.read_csv("experiment1.csv")

    T = np.unique(df["Year"]).shape[0]

    d15n=df.loc[ df["Year"]== 2015, ["Population"] ].values
    d16n=df.loc[ df["Year"]== 2016, ["Population"] ].values

    d15y=df.loc[ df["Year"]== 2015, ["Property crime"] ].values
    d16y=df.loc[ df["Year"]== 2016, ["Property crime"] ].values

    n=np.array([d15n, d16n]).squeeze(-1).T
    y=np.array([d15y, d16y]).squeeze(-1).T

    df["Regions"] = pd.Categorical(df["Regions"])
    df["Area"] = pd.Categorical(df["Area"])
    df["Year"] = pd.Categorical(df["Year"])

    region = df["Regions"].cat.codes.values.reshape(-1)
    year = df["Year"].cat.codes.values.reshape(-1)
    states = df["Area"].cat.codes.values.reshape(-1)

    nregions = np.unique(region).shape[0]
    nstates = n.shape[0]


    models = mcodes(
        region_prior, 
        state_prior, 
        n, 
        y, 
        (year, T), 
        (region, nregions), 
        (states, nstates) 
    ) 

    return models




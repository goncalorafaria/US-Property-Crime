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

def mcodes_subtypes(region_prior, state_prior, 
        yn, k, p, yeari, regioni, statei):

    Wr, isor, nisor, Dr, Ir, Wrt = region_prior 
    Ws, isos, nisos, Ds, Is, Wst = state_prior 

    year, T = yeari 
    region, nregions = regioni
    states, nstates = statei 

    models={}

    modelcode = '''
    model
    {
        #### Likelihood
                
        for (t in 1:T){
            for (s in 1:nstates) {    
                for( i in 1:k ){
                    logit(rate[s,t,i]) <- th[i] + sieffect[i,s,t]
                }
                
                p[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                y_pred[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                log.likelihood[s,t] <- logdensity.multi(p[s,t,1:k], rate[s,t,1:k], y[s,t])
                
                likelihood[s,t] <- exp(log.likelihood[s,t])
        }
        }

        log.likelihood_sum <- sum(log.likelihood)
        
        #### Priors
        
        for( i in 1:k ){ th.mu[i] ~ dnorm(0, 1.0E-3) } 
        
        for( i in 1:k ){ th.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 ) } 
            
        th ~ dmnorm( th.mu ,th.sigma)
        
        for( i in 1:k ){
            for( j in 1:k ){
                th.sigma[i,j] <- (i==j)* th.tau[i]
            }
        }
        
        #### AutoRegressive errors
        
        for( i in 1:k ){ 
            sieffect.mu[i] ~ dnorm(0, 1.0E-3);
            sieffect.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 );
            w[i] ~ dnorm(0, 1.0E-3);
            
            for( j in 1:k ){
                sieffect.sigma[i,j] <- (i==j)* sieffect.tau[i]
            }
            
        } 
        
        for( s in 1:nstates ){
            sieffect[1:k,s,1] ~ dmnorm( sieffect.mu ,sieffect.sigma)
            
            for( t in 2:T ){
                conststep[1:k,s,t] ~ dmnorm( sieffect.mu ,sieffect.sigma)
                for( i in 1:k){
                    sieffect[i,s,t] <- w[i] * (conststep[i,s,t] - sieffect[i,s,t-1]) + conststep[i,s,t]
                }
            }
        }
        
    }
    '''
    varnames = ['log.likelihood_sum','log.likelihood','likelihood','rate','y_pred','w']
    par = 3

    bvars = dict(T=T, nstates=nstates, k=k, y=yn, p=p)
        
    models["uI"] = (modelcode,varnames,bvars,par)

    #####

    modelcode = '''
    model
    {
        #### Likelihood
                
        for (t in 1:T){
            for (s in 1:nstates) {    
                for( i in 1:k ){
                    logit(rate[s,t,i]) <- th[i] + reffect[i, region[s]+1] + sieffect[i,s,t]
                }
                
                p[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                y_pred[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                log.likelihood[s,t] <- logdensity.multi(p[s,t,1:k], rate[s,t,1:k], y[s,t])
                
                likelihood[s,t] <- exp(log.likelihood[s,t])
        }
        }

        log.likelihood_sum <- sum(log.likelihood)
        
        #### Priors
        
        ## National effect
        
        for( i in 1:k ){ th.mu[i] ~ dnorm(0, 1.0E-3) } 
        
        for( i in 1:k ){ th.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 ) } 
            
        th ~ dmnorm( th.mu ,th.sigma)
        
        for( i in 1:k ){
            for( j in 1:k ){
                th.sigma[i,j] <- (i==j)* th.tau[i]
            }
        }
        
        #### Random Effects
        
        ## Regional Effect
        
        for(i in 1:k ){
            car.phi[i] ~ dunif(-0.99, 0.99)
            car.tau[i] ~ dgamma( 1.0E-3,  1.0E-3)
            car[i,1:length(nisor)] ~ dmnorm(rep(0, length(nisor)), car.tau[i] * Dr %*% (Ir - car.phi[i]*Wr))
            
            for (j in 1:length(nisor)) { reffect[i, nisor[j]+1] <- car[i,j] }
            for (j in 1:length(isor)) { reffect[i, isor[j]+1] ~ dnorm(0, car.tau[i]) }
        }    
        
        #### AutoRegressive errors
        
        for( i in 1:k ){ 
            sieffect.mu[i] ~ dnorm(0, 1.0E-3);
            sieffect.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 );
            w[i] ~ dnorm(0, 1.0E-3);
            
            for( j in 1:k ){
                sieffect.sigma[i,j] <- (i==j)* sieffect.tau[i]
            }
            
        } 
        
        for( s in 1:nstates ){
            sieffect[1:k,s,1] ~ dmnorm( sieffect.mu ,sieffect.sigma)
            
            for( t in 2:T ){
                conststep[1:k,s,t] ~ dmnorm( sieffect.mu ,sieffect.sigma)
                for( i in 1:k){
                    sieffect[i,s,t] <- w[i] * (conststep[i,s,t] - sieffect[i,s,t-1]) + conststep[i,s,t]
                }
            }
        }
        
    }
    '''
    varnames = ['log.likelihood_sum','log.likelihood','likelihood','rate','y_pred','w']
    par = 3

    bvars = dict(T=T, nstates=nstates, k=k, y=yn, p=p,region=region, 
                Ir=Ir, Dr=Dr, Wr=Wr, nisor=nisor, isor=isor)
        
    models["uII"] = (modelcode,varnames,bvars,par)

    ####
    modelcode = '''
    model
    {
        #### Likelihood
                
        for (t in 1:T){
            for (s in 1:nstates) {    
                for( i in 1:k ){
                    logit(rate[s,t,i]) <- th[i] + seffect[i,s] + sieffect[i,s,t]
                }
                
                p[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                y_pred[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                log.likelihood[s,t] <- logdensity.multi(p[s,t,1:k], rate[s,t,1:k], y[s,t])
                
                likelihood[s,t] <- exp(log.likelihood[s,t])
        }
        }

        log.likelihood_sum <- sum(log.likelihood)
        
        #### Priors
        
        ## National effect
        
        for( i in 1:k ){ 
            th.mu[i] ~ dnorm(0, 1.0E-3);
            th.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 ); 
            for( j in 1:k ){
                th.sigma[i,j] <- (i==j)* th.tau[i]
            }
        } 
            
        th ~ dmnorm( th.mu ,th.sigma)
        
        #### Random Effects
        ## State effect
        
        for( i in 1:k){
            cas.phi[i] ~ dunif(-0.99, 0.99)
            cas.tau[i] ~ dgamma( 1.0E-3,  1.0E-3)
            
            cas[i,1:length(nisos)] ~ dmnorm(rep(0, length(nisos)), cas.tau[i] * Ds %*% (Is - cas.phi[i]*Ws))

            for (j in 1:length(nisos)) { seffect[i,nisos[j]+1] <- cas[i,j] }
            
            for (j in 1:length(isos)) { seffect[i,isos[j]+1] ~ dnorm(0, cas.tau[i]) }
        } 
        
        #### AutoRegressive errors
        
        for( i in 1:k ){ 
            sieffect.mu[i] ~ dnorm(0, 1.0E-3);
            sieffect.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 );
            w[i] ~ dnorm(0, 1.0E-3);
            
            for( j in 1:k ){
                sieffect.sigma[i,j] <- (i==j) * sieffect.tau[i]
            }
            
        } 
        
        for( s in 1:nstates ){
            sieffect[1:k,s,1] ~ dmnorm( sieffect.mu, sieffect.sigma)
            
            for( t in 2:T ){
                conststep[1:k,s,t] ~ dmnorm( sieffect.mu ,sieffect.sigma)
                for( i in 1:k){
                    sieffect[i,s,t] <- w[i] * (conststep[i,s,t] - sieffect[i,s,t-1]) + conststep[i,s,t]
                }
            }
        }
        
    }
    '''
    varnames = ['log.likelihood_sum','log.likelihood','likelihood','rate','y_pred','w']
    par = 3
    bvars = dict(T=T, nstates=nstates, k=k, y=yn, p=p,
                Is=Is, Ds=Ds, Ws=Ws, nisos=nisos, isos=isos)
        
    models["uIII"] = (modelcode,varnames,bvars,par)

    ####

    modelcode = '''
    model
    {
        #### Likelihood
                
        for (t in 1:T){
            for (s in 1:nstates) {    
                for( i in 1:k ){
                    logit(rate[s,t,i]) <- th[i] + reffect[i, region[s]+1] + seffect[i,s] + sieffect[i,s,t]
                }
                
                p[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                y_pred[s,t,1:k] ~ dmulti(rate[s,t,1:k], y[s,t])
                
                log.likelihood[s,t] <- logdensity.multi(p[s,t,1:k], rate[s,t,1:k], y[s,t])
                
                likelihood[s,t] <- exp(log.likelihood[s,t])
        }
        }

        log.likelihood_sum <- sum(log.likelihood)
        
        #### Priors
        
        ## National effect
        
        for( i in 1:k ){ th.mu[i] ~ dnorm(0, 1.0E-3) } 
        
        for( i in 1:k ){ th.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 ) } 
            
        th ~ dmnorm( th.mu ,th.sigma)
        
        for( i in 1:k ){
            for( j in 1:k ){
                th.sigma[i,j] <- (i==j)* th.tau[i]
            }
        }
        
        #### Random Effects
        ## State effect
        
        for( i in 1:k){
            cas.phi[i] ~ dunif(-0.99, 0.99)
            cas.tau[i] ~ dgamma( 1.0E-3,  1.0E-3)
            
            cas[i,1:length(nisos)] ~ dmnorm(rep(0, length(nisos)), cas.tau[i] * Ds %*% (Is - cas.phi[i]*Ws))

            for (j in 1:length(nisos)) { seffect[i,nisos[j]+1] <- cas[i,j] }
            
            for (j in 1:length(isos)) { seffect[i,isos[j]+1] ~ dnorm(0, cas.tau[i]) }
        }
            
        ## Regional Effect
        
        for(i in 1:k ){
            car.phi[i] ~ dunif(-0.99, 0.99)
            car.tau[i] ~ dgamma( 1.0E-3,  1.0E-3)
            car[i,1:length(nisor)] ~ dmnorm(rep(0, length(nisor)), car.tau[i] * Dr %*% (Ir - car.phi[i]*Wr))
            
            for (j in 1:length(nisor)) { reffect[i, nisor[j]+1] <- car[i,j] }
            for (j in 1:length(isor)) { reffect[i, isor[j]+1] ~ dnorm(0, car.tau[i]) }
        }    
        
        #### AutoRegressive errors
        
        for( i in 1:k ){ 
            sieffect.mu[i] ~ dnorm(0, 1.0E-3);
            sieffect.tau[i] ~ dgamma( 1.0E-3, 1.0E-3 );
            w[i] ~ dnorm(0, 1.0E-3);
            
            for( j in 1:k ){
                sieffect.sigma[i,j] <- (i==j)* sieffect.tau[i]
            }
            
        } 
        
        for( s in 1:nstates ){
            sieffect[1:k,s,1] ~ dmnorm( sieffect.mu ,sieffect.sigma)
            
            for( t in 2:T ){
                conststep[1:k,s,t] ~ dmnorm( sieffect.mu ,sieffect.sigma)
                for( i in 1:k){
                    sieffect[i,s,t] <- w[i] * (conststep[i,s,t] - sieffect[i,s,t-1]) + conststep[i,s,t]
                }
            }
        }
        
    }
    '''
    varnames = ['log.likelihood_sum','log.likelihood','likelihood','rate','y_pred','w']

    par = 3

    bvars = dict(T=T, nstates=nstates, k=k, y=yn, p=p,region=region, 
                Ir=Ir, Dr=Dr, Wr=Wr, nisor=nisor, isor=isor,
                Is=Is, Ds=Ds, Ws=Ws, nisos=nisos, isos=isos)
        
    models["uIV"] = (modelcode,varnames,bvars,par)

    return models

def mcodes(region_prior,state_prior,n , y, yeari, regioni, statei):

    Wr, isor, nisor, Dr, Ir, Wrt = region_prior 
    Ws, isos, nisos, Ds, Is, Wst = state_prior 

    year, T = yeari 
    region, nregions = regioni
    states, nstates = statei 

    models={}
    ######
    #####


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

                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])

            }
        }

        log.lik <- sum(log.likelihood)

        #### Parameters: ############### 
        # Priors
        #--- Time
        t.c ~ dnorm(0,1.0e-3)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 
    
        ##--- regional sas
    
        for(s in 1:(nregions) ){ reffect[s] ~ dnorm(0, 1.0E-3) } 
        
        # --- Epsilon

        ## iid part

        w ~ dnorm(0, 1e-2)

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        } 
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) } 
    }
    '''

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','t.c','w','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,
    region=region, nregions=nregions, y=y, n=n, T=T)
    models["IVi"] = (modelcode,varnames,bvars,par)


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

                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])

            }
        }

        log.lik <- sum(log.likelihood)

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

        w ~ dnorm(0, 1e-2)

        for(s in 1:(nstates) ){ sieffect[s,1] ~ dnorm(0, 1.0E-3) } 

        for(t in 2:T){
            for(s in 1:(nstates) ){ 
                consteps[s,t] ~ dnorm(0, 1.0E-3);
                sieffect[s,t] <-  (sieffect[s,t-1] - consteps[s,t]) * w + consteps[s,t];
            }  
        } 

        ## auto regressive errors

        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) } 
    }
    '''

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','w','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,y=y, n=n, T=T)

    models["IIIi"] = (modelcode,varnames,bvars,par)

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
                
                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])
            }
        }

        log.lik <- sum(log.likelihood)

        #### Parameters: ############### 
        # Priors
        #--- Time
        
        w ~ dnorm(0,1.0e-3)
        
        #--- Nation
        neffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        neffect.beta ~ dnorm(0, 1.0E-3)

        neffect ~ dnorm(neffect.beta, neffect.tau) 
    
        ##--- regional sas
    
        for(s in 1:(nregions) ){ reffect[s] ~ dnorm(0, 1.0E-3) } 

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

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,
    region=region,
    nregions=nregions, y=y, n=n, T=T)
    models["IIi"] = (modelcode,varnames,bvars,par)

    ###

    #####
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

                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])

            }
        }

        log.lik <- sum(log.likelihood)

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

        w ~ dnorm(0, 1e-2)

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

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','t.c','w','sar.phi','sas.phi','lambda']

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

                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])

            }
        }

        log.lik <- sum(log.likelihood)

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

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','w','sas.phi','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,y=y, n=n, T=T, 
    isos=isos, nisos=nisos, Is=Is, Ds=Ds,Ws=Ws)

    models["III"] = (modelcode,varnames,bvars,par)

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
                
                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])
            }
        }

        log.lik <- sum(log.likelihood)

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

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','sar.phi','lambda','sar.mu','sar']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,
    region=region, y=y, n=n, T=T, 
    isor=isor, nisor=nisor, Ir=Ir, Dr=Dr, Wr=Wr)
    models["II"] = (modelcode,varnames,bvars,par)

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

                likelihood[s,t] <- dpois(y[s,t], lambda[s,t])
                log.likelihood[s,t] <- log(likelihood[s,t])
            }
        }

        log.lik <- sum(log.likelihood)

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

    varnames = ['log.likelihood', 'likelihood','y_pred','theta','log.lik','w','lambda']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates,y=y, n=n, T=T)
    
    models["I"] = (modelcode,varnames,bvars,par)

    ###
    return models

def build_propertycrime(seed = 123):

    np.random.seed(seed)

    df = pd.read_csv("experiment1.csv")

    region_prior = build_region_proximity_matrix()
    state_prior = build_state_proximity_matrix()

    T = np.unique(df["Year"]).shape[0]

    d15n=df.loc[ df["Year"]== 2015, ["Population"] ].values
    d16n=df.loc[ df["Year"]== 2016, ["Population"] ].values

    d15y=df.loc[ df["Year"]== 2015, ["Property crime"] ].values
    d16y=df.loc[ df["Year"]== 2016, ["Property crime"] ].values

    n=np.array([d15n, d16n]).squeeze(-1).T
    y=np.array([d15y, d16y]).squeeze(-1).T

    df["Area"] = pd.Categorical(df["Area"])
    df["Regions"] = pd.Categorical(df["Regions"])
    df["Year"] = pd.Categorical(df["Year"])

    states = df["Area"].cat.codes.values.reshape(-1)
    region = df["Regions"].cat.codes.values.reshape(-1)
    year = df["Year"].cat.codes.values.reshape(-1)

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

def build_subtypes(seed = 123):

    np.random.seed(seed)

    df = pd.read_csv("experiment1.csv")
    
    region_prior = build_region_proximity_matrix()
    state_prior = build_state_proximity_matrix()

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
    b=np.array([d15y, d16b]).squeeze(-1).T
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

    models = mcodes_subtypes(
        region_prior, 
        state_prior, 
        yn,
        k,
        p, 
        (year, T), 
        (region, nregions), 
        (states, nstates) 
    ) 

    return models




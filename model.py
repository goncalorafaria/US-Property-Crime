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
    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p
                            
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])

            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
    }
    '''

    varnames = ['y_pred','theta','log.lik']
    par = 3
    bvars = dict(nstates = nstates, y = y, n = n, T=T)

    models["fiis"] = (modelcode,varnames,bvars,par)
    # In[164]:
    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p + ireffect[region[s]+1]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois( theta[s,t] * n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t] * n[s,t])

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
                
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- iid Region
        
        ireffect.tau ~ dgamma( 1.0E-3,  1.0E-3)
        ireffect.beta ~ dnorm(0, 1.0E-3)
        
        for (i in 1:nregions) { ireffect[i] ~ dnorm(ireffect.beta, ireffect.tau) }
        
    }
    '''

    varnames = ['y_pred','theta','log.lik']
    par = 15
    bvars = dict(nstates = nstates, y = y, n = n, nregions=nregions, region=region, T=T)
    models["ffis"] = (modelcode,varnames,bvars,par)

    # In[187]:
    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p + reffect[region[s]+1]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois( theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])
                
                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
                
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- CAR Region
        
        car.phi ~ dunif(-0.99, 0.99)
        car.tau ~ dgamma( 1.0E-3,  1.0E-3)
                
        car ~ dmnorm(rep(0, length(nisor), car.tau * Dr %*% (Ir - car.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- car[i] }
        
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, car.tau) }
            
    }
    '''

    varnames = ['y_pred','theta','log.lik']

    par = 15

    bvars = dict(
        nstates = nstates, 
        y = y, 
        n = n, 
        Dr=Dr, 
        Wr=Wr,
        Ir=Ir,
        isor=isor,
        nisor=nisor, 
        region=region, 
        T=T)

    models["fcis"] = (modelcode,varnames,bvars,par)

    
    # In[189]:
    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }

        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p + reffect[region[s]+1]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2] * eps[s,t]

                theta[s,t] <-  mu.effect[s,t] * mu.t[s,t]
                
                lambda[s,t] <- theta[s,t]* n[s,t]

                y[s,t] ~ dpois(lambda[s,t])
                y_pred[s,t] ~ dpois(lambda[s,t])

                plog.lik[s,t] <- logdensity.pois(y[s,t], lambda[s,t])

                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)

        #--- Time
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }

        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- SAR Region
        
        sar.phi ~ dunif(-0.99, 0.99)
        sar.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        for(i in 1:length(nisor) ){ sar.beta[i] <- 0 }
        
        sar ~ dmnorm(sar.beta, sar.tau * (Ir - sar.phi*Wrt) %*% Dr %*% t(Ir - sar.phi*Wrt) )
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- sar[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, sar.tau) }
        
    }
    '''

    varnames = ['y_pred','theta','log.lik']

    par = 15

    bvars = dict(nstates = nstates, y = y, n = n, nisor=nisor,isor=isor, Dr=Dr,Ir=Ir, Wrt=Wrt, region=region, T=T)
    models["fsil"] = (modelcode,varnames,bvars,par)

    
    
    # In[189]:
    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p + reffect[region[s]+1]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois( theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- SAR Region
        
        sar.phi ~ dunif(-0.99, 0.99)
        sar.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        for(i in 1:length(nisor) ){ sar.beta[i] <- 0 }
        
        sar ~ dmnorm(sar.beta, sar.tau * (Ir - sar.phi*Wrt) %*% Dr %*% t(Ir - sar.phi*Wrt) )
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- sar[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, sar.tau) }
        
    }
    '''

    varnames = ['y_pred','theta','log.lik']

    par = 15

    bvars = dict(nstates = nstates, y = y, n = n, nisor=nisor,isor=isor, Dr=Dr,Ir=Ir, Wrt=Wrt, region=region, T=T)
    models["fsis"] = (modelcode,varnames,bvars,par)

    # In[195]:

    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + reffect[region[s]+1] + seffect[s]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])    

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t]) 
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }
        
        #--- CAR Region
        
        car.phi ~ dunif(-0.99, 0.99)
        car.tau ~ dgamma( 1.0E-3,  1.0E-3)
                
        car ~ dmnorm(rep(0, length(nisor), car.tau * Dr %*% (Ir - car.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- car[i] }
        
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, car.tau) }
        
    }
    '''

    par = 15 + 52

    varnames = ['y_pred','theta','log.lik']

    bvars = dict(nstates = nstates, y = y, isor=isor,nisor=nisor ,n = n, Dr=Dr, Wr=Wr, Ir=Ir, region=region, T=T)
    models["fcfs"] = (modelcode,varnames,bvars,par)
    # In[193]:

    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + reffect[region[s]+1] + seffect[s]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])     

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm( 0, 1.0E-5)
        p ~ dnorm(p.beta, p.tau) 
        
        #--- CAR State
        
        cas.phi ~ dunif(-0.99, 0.99)

        cas.tau ~ dgamma( 1.0E-1,  1.0E-1)
                
        cas ~ dmnorm(rep(0, length(nisos), cas.tau * Ds %*% (Is - cas.phi*Ws))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- cas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, cas.tau) }
        
        #--- CAR Region
        
        car.phi ~ dunif(-0.99, 0.99)
        car.tau ~ dgamma( 1.0E-1,  1.0E-1)
                
        car ~ dmnorm(rep(0, length(nisor), car.tau * Dr %*% (Ir - car.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- car[i] }
        
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, car.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','car.phi','cas.phi','car.tau','cas.tau','p','log.lik']

    bvars = dict(nstates = nstates, y = y, isor=isor,nisor=nisor ,n = n, Dr=Dr, Wr=Wr, Ir=Ir, Ws=Ws, Ds=Ds, Is=Is, isos=isos, nisos=nisos, region=region, T=T)
    models["fccs"] = (modelcode,varnames,bvars,par)
    
    ######

    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])     

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }

        log.lik <- mean(plog.lik)
        
        # Priors
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- CAR State
        
        cas.phi ~ dunif(-0.99, 0.99)
        cas.tau ~ dgamma( 1.0E-3,  1.0E-3)
                
        cas ~ dmnorm(rep(0, length(nisos), cas.tau * Ds %*% (Is - cas.phi*Ws))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- cas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, cas.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','cas.phi','cas.tau','p','log.lik']

    bvars = dict(nstates = nstates, y = y,n = n, Ws=Ws, Ds=Ds,Is=Is, isos=isos, nisos=nisos,T=T)
    models["fics"] = (modelcode,varnames,bvars,par)

    ######

    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }

        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2] * eps[s,t] 

                theta[s,t] <-  mu.effect[s,t] * mu.t[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])     

                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }

        log.lik <- mean(plog.lik)
        
        # Priors

        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- CAR State
        
        cas.phi ~ dunif(-0.99, 0.99)
        cas.tau ~ dgamma( 1.0E-3,  1.0E-3)
                
        cas ~ dmnorm(rep(0, length(nisos)), cas.tau * Ds %*% (Is - cas.phi*Ws))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- cas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, cas.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','cas.phi','cas.tau','p','log.lik']

    bvars = dict(nstates = nstates, y = y,n = n, Ws=Ws, Ds=Ds,Is=Is, isos=isos, nisos=nisos,T=T)
    models["ficl"] = (modelcode,varnames,bvars,par)



    ####


    modelcode = '''
    model
    {
        # Likelihood
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t]* n[s,t] )
                y_pred[s,t] ~ dpois(theta[s,t]* n[s,t])     

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }

        log.lik <- mean(plog.lik)
        
        # Priors
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- SAR State
        
        sas.phi ~ dunif(-0.99, 0.99)
        sas.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        for(i in 1:length(nisos) ){ sas.beta[i] <- 0 }
        
        sas ~ dmnorm(sas.beta, sas.tau * (Is - sas.phi*Wst) %*% Ds %*% t(Is - sas.phi*Wst))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- sas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, sas.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','sas.phi','sas.tau','p','log.lik']

    bvars = dict(nstates = nstates, y = y,n = n, Wst=Wst, Ds=Ds, Is= Is, isos=isos, nisos=nisos,T=T)
    models["fiss"] = (modelcode,varnames,bvars,par)


    ####

    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + reffect[region[s]+1] + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2]* eps[s,t] 
                
                theta[s,t] <-  mu.effect[s,t]* mu.t[s,t]
                
                y[s,t] ~ dpois( theta[s,t] * n[s,t] )
                y_pred[s,t] ~ dpois( theta[s,t] * n[s,t])
                
                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors
        
        log.lik <- mean(plog.lik)

        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }
        
        #--- CAR Region
        
        car.phi ~ dunif(-0.99, 0.99)
        car.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        car ~ dmnorm(rep(0, length(nisor), car.tau * Dr %*% (Ir - car.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- car[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, car.tau) }
        
        
    }
    '''

    varnames = ['y_pred','t.beta','theta','log.lik']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates, y = y, n = n, nisor=nisor, isor=isor, Dr=Dr, Wr=Wr,Ir=Ir, region=region, T=T)
    models["fcfl"] = (modelcode,varnames,bvars,par)

    ###

    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + reffect[region[s]+1] + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2]* eps[s,t] 
                
                theta[s,t] <-  mu.effect[s,t]* mu.t[s,t]
                
                y[s,t] ~ dpois( theta[s,t] * n[s,t] )
                y_pred[s,t] ~ dpois( theta[s,t] * n[s,t])
                
                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors
        
        log.lik <- mean(plog.lik)

        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }

        for(r in 1:(nregions) ){ reffect[r] ~ dnorm(0, 1.0E-3) }
        
        
    }
    '''

    varnames = ['y_pred','t.beta','theta','log.lik']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates, nregions=nregions, region=region, y=y, n=n, T=T)
    models["fffl"] = (modelcode,varnames,bvars,par)

    # In[192]:


    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2] * eps[s,t]
                
                theta[s,t] <- mu.effect[s,t] * mu.t[s,t]
                
                y[s,t] ~ dpois(theta[s,t] * n[s,t])
                y_pred[s,t] ~ dpois(theta[s,t] * n[s,t])
                
                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors
        
        log.lik <- mean(plog.lik)

        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }

    }
    '''

    varnames = ['y_pred','t.beta','theta','log.lik']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates, y=y, n=n, T=T)
    models["fifl"] = (modelcode,varnames,bvars,par)


     # In[192]:


    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                
                theta[s,t] <-  mu.effect[s,t]
                
                y[s,t] ~ dpois( theta[s,t] * n[s,t] )
                y_pred[s,t] ~ dpois( theta[s,t] * n[s,t])
                
                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors
        
        log.lik <- mean(plog.lik)
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }

    }
    '''

    varnames = ['y_pred','theta','log.lik']

    par = 15 + 52 + 2

    bvars = dict(nstates = nstates, y=y, n=n, T=T)
    models["fifs"] = (modelcode,varnames,bvars,par)

    ###

    modelcode = '''
    model
    {
        # Likelihood
        
        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + reffect[region[s]+1] + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2]* eps[s,t] + t.beta[3]* eps[s,t]^2
                
                theta[s,t] <-  mu.effect[s,t] * mu.t[s,t]
                
                y[s,t] ~ dpois(theta[s,t] * n[s,t] )
                y_pred[s,t] ~ dpois( theta[s,t]* n[s,t])  
                
                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }
        
        # Priors

        log.lik <- mean(plog.lik)
        
        #--- Time
        
        for(i in 1:3){ t.beta[i] ~ dnorm(0, 1.0E-3) }
        
        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- State iid
        
        for(s in 1:(nstates) ){ seffect[s] ~ dnorm(0, 1.0E-3) }
        
        #--- CAR Region
        
        car.phi ~ dunif(-0.99, 0.99)
        car.tau ~ dgamma( 1.0E-3,  1.0E-3)
                
        car ~ dmnorm(rep(0, length(nisor), car.tau * Dr %*% (Ir - car.phi*Wr))
        
        for (i in 1:length(nisor)) { reffect[nisor[i]+1] <- car[i] }
        for (i in 1:length(isor)) { reffect[isor[i]+1] ~ dnorm(0, car.tau) }
        
        
    }
    '''

    varnames = ['y_pred','t.beta','theta','log.lik']

    par = 15 + 52 + 3

    bvars = dict(nstates = nstates, y = y, n = n,nisor=nisor , isor=isor,Dr=Dr, Wr=Wr,Ir=Ir, region=region, T=T)
    models["fcfq"] = (modelcode,varnames,bvars,par)
    
    #

    modelcode = '''
    model
    {
        # Likelihood

        for(s in 1:nstates){ eps[s,1] <- 0 }
        
        for (t in 1:T){
            for (s in 1:nstates) {
                log(mu.effect[s,t]) <- p + seffect[s]
                log(mu.t[s,t]) <- t.beta[1] + t.beta[2]* eps[s,t] 

                theta[s,t] <-  mu.t[s,t] * mu.effect[s,t]
                
                y[s,t] ~ dpois(theta[s,t] * n[s,t] )
                
                y_pred[s,t] ~ dpois(theta[s,t] * n[s,t])     

                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }

        #
        mse <- mean(eps)
        log.lik <- mean(plog.lik)
        
        # Priors
        
        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }

        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- SAR State
        
        sas.phi ~ dunif(-0.99, 0.99)
        sas.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        for(i in 1:length(nisos) ){ sas.beta[i] <- 0 }
        
        sas ~ dmnorm(sas.beta, sas.tau * (Is - sas.phi*Wst) %*% Ds %*% t(Is - sas.phi*Wst))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- sas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, sas.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','mse','log.lik']

    bvars = dict(nstates = nstates, y = y,n = n, Wst=Wst, Ds=Ds, Is= Is, isos=isos, nisos=nisos,T=T)
    models["fisl"] = (modelcode,varnames,bvars,par)

    ###

    modelcode = '''
    model
    {
        # Likelihood

        for(s in 1:nstates){ eps[s,1] <- 0 }

        for (t in 1:T){
            for (s in 1:nstates) {
                logit(mu.effect[s,t]) <- p + seffect[s]
                logit(mu.t[s,t]) <- t.beta[1] + t.beta[2] * eps[s,t] 

                theta[s,t] <-  mu.t[s,t] * mu.effect[s,t]
                
                y[s,t] ~ dnegbin( theta[s,t], n[s,t] )

                y_pred[s,t] ~ dnegbin( theta[s,t], n[s,t] )     

                eps[s,t+1] <- log( ( (y[s,t] - y_pred[s,t]) / y_pred[s,t]) + 1 )

                plog.lik[s,t] <- logdensity.pois(y[s,t], theta[s,t] * n[s,t])
            }
        }

        #
        mse <- mean(eps)
        log.lik <- mean(plog.lik)
        
        # Priors
        
        #--- Time
        
        for(i in 1:2){ t.beta[i] ~ dnorm(0, 1.0E-3) }

        #--- Nation
        p.tau ~ dgamma( 1.0E-3,  1.0E-3)
        p.beta ~ dnorm(0, 1.0E-3)

        p ~ dnorm(p.beta, p.tau) 
        
        #--- SAR State
        
        sas.phi ~ dunif(-0.99, 0.99)
        sas.tau ~ dgamma( 1.0E-3,  1.0E-3)
        
        for(i in 1:length(nisos) ){ sas.beta[i] ~ dnorm(0, 0.01) }
        
        sas ~ dmnorm(sas.beta, sas.tau * (Is - sas.phi*Wst) %*% Ds %*% t(Is - sas.phi*Wst))
        
        for (i in 1:length(nisos)) { seffect[nisos[i]+1] <- sas[i] }
        
        for (i in 1:length(isos)) { seffect[isos[i]+1] ~ dnorm(0, sas.tau) }
        
    }
    '''

    par = 15 + 52 + 2

    varnames = ['y_pred','theta','mse','log.lik']

    bvars = dict(nstates = nstates, y = y,n = n, Wst=Wst, Ds=Ds, Is= Is, isos=isos, nisos=nisos,T=T)
    models["xfisl"] = (modelcode,varnames,bvars,par)

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




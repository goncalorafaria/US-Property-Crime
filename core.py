"""
# Header ------------------------------------------------------------------
"""
import pyjags
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Jags code ---------------------------------------------------------------

# Jags code to fit the model to the simulated data

def sample(code, bvars ,varnames,chains=4,trials=10000):
    # Set up the data:
    model = pyjags.Model(code, data=bvars, chains=4) # Chains = Number of different starting positions
    # Choose the parameters to watch and iterations:
    samples = model.sample(trials, vars=varnames)
    
    return samples

def summary(samples, varname, p=95):
    values = samples[varname]
    ci = np.percentile(values, [100-p, p])
    print(varname + ' mean = ' + str(np.mean(values, axis=(1,2))) + ', ' + str(p) + '% credible interval ' + str(ci) )

def plot(trace, var, ind):
    if( trace[var].shape[0] > 1):
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(var, fontsize='xx-large')
    #print(var)
    #print(trace.shape)
    #print(trace)
    s = pd.Series(trace[var][:,:].reshape(-1))
    #ax = s.plot.kde()
    
    # Marginal posterior density estimate:
    axes[0].hist(s, color = 'blue', edgecolor = 'black',
             bins = 100)
    plt.title('Marginal')

    # Autocorrelation for each chain:
    #axes[1].set_xlim(0, 100)
    for chain in range(trace[list(ind.keys())[0]].shape[-1]):
        pd.plotting.autocorrelation_plot(trace[var][:,:,chain].reshape(-1), axes[1], label=chain)

    ## value trace plot:
    axes[2].plot(
        list(
            range(
                len(
                    trace[var].reshape(-1)
                )
            )
        ),
        trace[var].reshape(-1)
    )
    
def analyse_fit(varnames,samples):
    for varname in varnames:
        summary(samples, varname)
    
    i=0
    ind ={}
    for k in samples.keys():
        ind[k] = i
        i+=1
    # Display diagnostic plots
    for var in samples.keys():
        plot(samples, var, ind)
    
    plt.show()
    
    if ("alpha" in samples) and ("beta" in samples):
        np.mean(samples["alpha"])
        np.mean(samples["beta"])
        prior = np.random.beta(np.mean(samples["alpha"]), np.mean(samples["beta"]), size=4000)
    
        plt.hist(prior)

def plot_p(trace, var, ind):
    
    samples = trace
    tvar = var
    for i in range(samples[var].shape[0]):
        var = var + str(i)
        
        trace[var] = samples[tvar][i]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        #fig.suptitle(var, fontsize='xx-large')
        #print(var)
        #print(trace.shape)
        #print(trace)
        s = pd.Series(trace[var][:,:].reshape(-1))
        #ax = s.plot.kde()

        # Marginal posterior density estimate:
        axes[0].hist(s, color = 'blue', edgecolor = 'black',
                 bins = 100)
        plt.title('Marginal')

        # Autocorrelation for each chain:
        #axes[1].set_xlim(0, 100)
        for chain in range(trace[list(ind.keys())[0]].shape[-1]):
            pd.plotting.autocorrelation_plot(trace[var][:,chain].reshape(-1), axes[1], label=chain)

        ## value trace plot:
        axes[2].plot(
            list(
                range(
                    len(
                        trace[var].reshape(-1)
                    )
                )
            ),
            trace[var].reshape(-1)
        )
    
def analyse_fit_pred(varnames,samples):
    for varname in varnames:
        summary(samples, varname)
    
    i=0
    ind ={}
    for k in samples.keys():
        ind[k] = i
        i+=1
    # Display diagnostic plots
    for var in samples.keys():
        plot_p(samples, var, ind)
    
    plt.show()
        
# Create a plot of the posterior mean:

# Creating a plot:
def plot_predictive(x, y, y_pred):
    plt.scatter(x,y, c="blue", label="Data")
    plt.ylabel('y')
    plt.xlabel('x')
    #plt.plot(x,y, c= "red", label = "Truth")
    
    #sorted(range(len(s)), key=lambda k: s[k])
    
    plt.scatter(x, y_pred, c = "red",label = "Posterior")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    
    mse = np.sum(((y-y_pred)/y)**2) 
    print("chi-square : " + str(mse))
    
    
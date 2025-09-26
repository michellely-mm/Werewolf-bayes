import numpy as np
import matplotlib.pyplot as plt

def bayesian_update(prior, tau, n_updates = 4):
    """
    Use Bayes' thrm to update the posterior probability
    prior: the initial prior probability P(Wolf)
    tau: the probability that the player tells the truths
    n_updates: the number of times we observe the statement "I am not the wolf"
    """
    posteriors = [prior]
    p_wolf = prior
        #p_wolf records the belif of him being the wolf. It will be updated in each step
    for i in range(n_updates):
        P_S_given_W = 1 - tau
        P_S_given_V = tau
        
        numerator = P_S_given_W * p_wolf
        denominator = numerator + P_S_given_V * (1 - p_wolf)
        p_wolf = numerator / denominator
        
        posteriors.append(p_wolf)
        print(f'After statement {i+1}: P(Wolf|S) = {p_wolf:.4f}')
        
    return posteriors
    
if __name__ == "__main__": #will not show up if only import the file
    prior = 0.25
    tau = 0.7
    n_updates = 4
    
    posteriors = bayesian_update(prior, tau, n_updates)
    
    plt.plot(range(n_updates + 1), posteriors, marker = 'o')
    plt.title(f'Posterior Probability of Being Wolf (tau = {tau})')
    plt.xlabel('number of statements')
    plt.ylabel('P(Wolf|S)')
    plt.grid(True)
    plt.savefig('posterior_demo_tau0.7.png',dpi = 300)
    plt.show


# https://fehiepsi.github.io/blog/sampling-hmm-pyro/

import matplotlib.pyplot as plt
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS

pyro.set_rng_seed(1)

num_categories = 2
num_words = 10
num_supervised_data = 50
num_data = 100

transition_prior = torch.empty(num_categories).fill_(1.)
emission_prior = torch.empty(num_words).fill_(0.1)

transition_prob = dist.Dirichlet(transition_prior).sample(torch.Size([num_categories]))
emission_prob = dist.Dirichlet(emission_prior).sample(torch.Size([num_categories]))
print(transition_prob)
print(emission_prob)

def equilibrium(mc_matrix):
    n = mc_matrix.size(0)
    return (torch.eye(n) - mc_matrix.t() + 1).inverse().matmul(torch.ones(n))

start_prob = equilibrium(transition_prob)

# simulate data
categories, words = [], []
for t in range(num_data):
    if t == 0 or t == num_supervised_data:
        category = dist.Categorical(start_prob).sample()
    else:
        category = dist.Categorical(transition_prob[category]).sample()
    word = dist.Categorical(emission_prob[category]).sample()
    categories.append(category)
    words.append(word)
categories, words = torch.stack(categories), torch.stack(words)

# split into supervised data and unsupervised data
supervised_categories = categories[:num_supervised_data]
supervised_words = words[:num_supervised_data]
unsupervised_words = words[num_supervised_data:]

def forward_log_prob(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob = emission_log_prob[:, curr_word] + transition_log_prob + prev_log_prob.unsqueeze(dim=1)
    return log_prob.logsumexp(dim=0)

def unsupervised_hmm(words):
    with pyro.plate("prob_plate", num_categories):
        transition_prob = pyro.sample("transition_prob", dist.Dirichlet(transition_prior))
        emission_prob = pyro.sample("emission_prob", dist.Dirichlet(emission_prior))

    transition_log_prob = transition_prob.log()
    emission_log_prob = emission_prob.log()
    log_prob = emission_log_prob[:, words[0]]
    for t in range(1, len(words)):
        log_prob = forward_log_prob(log_prob, words[t], transition_log_prob, emission_log_prob)
    prob = log_prob.logsumexp(dim=0).exp()
    # a trick to inject an additional log_prob into model's log_prob
    pyro.sample("forward_prob", dist.Bernoulli(prob), obs=torch.tensor(1.))
nuts_kernel = NUTS(unsupervised_hmm, jit_compile=True, ignore_jit_warnings=True)
mcmc = MCMC(nuts_kernel, num_samples=100)
mcmc.run(unsupervised_words)
trace_transition_prob = mcmc.get_samples()["transition_prob"]
print(trace_transition_prob)


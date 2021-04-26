
# https://fehiepsi.github.io/blog/sampling-hmm-pyro/ translated into numpyro

import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from jax import jit, random
from jax.scipy.special import logsumexp

numpyro.set_host_device_count(2)

rng_key = random.PRNGKey(0)

num_categories = 2
num_words = 10
num_supervised_data = 50
num_data = 100

transition_prior = jnp.ones(num_categories) 
emission_prior = jnp.ones(num_words) * 0.1

transition_prob = dist.Dirichlet(transition_prior).sample(rng_key, (num_categories, ))
emission_prob = dist.Dirichlet(emission_prior).sample(rng_key, (num_categories, ))
print(transition_prob)
print(emission_prob)

def inv(x):
  return jnp.linalg.inv(x)

def equilibrium(mc_matrix):
    n = mc_matrix.shape[0]
    return jnp.matmul(inv(jnp.eye(n) - mc_matrix.transpose() + 1), jnp.ones(n))

start_prob = equilibrium(transition_prob)

# simulate data
categories, words = [], []
for t in range(num_data):
    if t == 0 or t == num_supervised_data:
        category = dist.Categorical(start_prob).sample(rng_key)
    else:
        category = dist.Categorical(transition_prob[category]).sample(rng_key)
    word = dist.Categorical(emission_prob[category]).sample(rng_key)
    categories.append(category)
    words.append(word)
categories, words = jnp.stack(categories), jnp.stack(words)

# split into supervised data and unsupervised data
supervised_categories = categories[:num_supervised_data]
supervised_words = words[:num_supervised_data]
unsupervised_words = words[num_supervised_data:]

def forward_log_prob(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob = emission_log_prob[:, curr_word] + transition_log_prob + jnp.expand_dims(prev_log_prob, 1)
    return logsumexp(log_prob, 0)

def unsupervised_hmm(words):
    with numpyro.plate("prob_plate", num_categories):
        transition_prob = numpyro.sample("transition_prob", dist.Dirichlet(transition_prior))
        emission_prob = numpyro.sample("emission_prob", dist.Dirichlet(emission_prior))

    transition_log_prob = jnp.log(transition_prob)
    emission_log_prob = jnp.log(emission_prob)
    log_prob = emission_log_prob[:, words[0]]
    for t in range(1, len(words)):
        log_prob = forward_log_prob(log_prob, words[t], transition_log_prob, emission_log_prob)
    prob = jnp.exp(logsumexp(log_prob, 0))
    # a trick to inject an additional log_prob into model's log_prob
    numpyro.sample("forward_prob", dist.Bernoulli(prob), obs=jnp.array(1.))
nuts_kernel = NUTS(unsupervised_hmm)
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100, num_chains=2)
mcmc.run(rng_key, words=unsupervised_words)
trace_transition_prob = mcmc.get_samples()["transition_prob"]
print(trace_transition_prob)


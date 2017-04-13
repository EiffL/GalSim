import theano
import theano.tensor as T
from theano.tensor.nnet import binary_crossentropy

import math

from .distributions import kl_normal2_stdnormal, log_normal2, kl_normal2_gm2, log_bernoulli, log_gm2
from .distributions import log_stdnormal

def iaf_gmm_gaussian_lower_bound(log_qz_given_x, pz_mu,  pz_log_var, pz_w, z,
                                 x_mu, x_log_var, x):
    """
    VAE loss function when the code follows a Gaussian Mixture prior and a
    posterior parametrised using IAF. The decoder being Gaussian
    """
    log_pz = log_gm2(z, mean=pz_mu, log_var=pz_log_var, weights=pz_w)
    log_px_given_z = log_normal2(x, x_mu, x_log_var, eps=1e-7).sum(axis=1)
    return log_pz + log_px_given_z - log_qz_given_x, log_qz_given_x -log_pz, log_px_given_z

def iaf_gmm_bernoulli_lower_bound(log_qz_given_x, pz_mu,  pz_log_var, pz_w, z,
                                  x_mu, x):
    """
    VAE loss function when the code follows a Gaussian Mixture prior and a
    posterior parametrised using IAF. The decoder being bernoulli
    """
    log_pz = log_gm2(z, mean=pz_mu, log_var=pz_log_var, weights=pz_w)
    log_px_given_z = log_bernoulli(x, x_mu, eps=1e-7).sum(axis=1)
    return log_pz + log_px_given_z - log_qz_given_x, log_qz_given_x -log_pz, log_px_given_z

def iaf_normal_bernoulli_lower_bound(log_qz_given_x, z,
                                  x_mu, x):
    """
    VAE loss function when the code follows a Gaussian Mixture prior and a
    posterior parametrised using IAF. The decoder being bernoulli
    """
    log_pz = log_stdnormal(z).sum(axis=1)
    log_px_given_z = log_bernoulli(x, x_mu, eps=1e-7).sum(axis=1)
    return log_pz + log_px_given_z - log_qz_given_x, log_pz, log_qz_given_x, log_px_given_z


def iaf_normal_gaussian_lower_bound(log_qz_given_x, z,
                                 x_mu, x_log_var, x):
    """
    VAE loss function when the code follows a Gaussian Mixture prior and a
    posterior parametrised using IAF. The decoder being Gaussian
    """
    log_pz = log_stdnormal(z).sum(axis=1)
    log_px_given_z = log_normal2(x, x_mu, x_log_var, eps=1e-7).sum(axis=1)
    return log_pz + log_px_given_z - log_qz_given_x, log_pz, log_qz_given_x, log_px_given_z


def normal_gaussian_lower_bound(z_mu, z_log_var, x_mu, x_log_var, x):
    """
    Variational Auto-Encoder loss function for gaussian code and output, under
    a normal prior.
    Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
    Latent z       : gaussian with standard normal prior
    Decoder output : gaussian
    """
    kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
    log_px_given_z = log_normal2(x, x_mu, x_log_var, eps=1e-7).sum(axis=1)
    return -kl_term + log_px_given_z, kl_term, log_px_given_z


def normal_bernoulli_lower_bound(z_mu, z_log_var, x_mu, x):
    """
    Variational Auto-Encoder loss function for gaussian code and bernoulli output, under
    a normal prior.
    Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
    Latent z       : gaussian with standard normal prior
    Decoder output : gaussian
    """
    kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
    log_px_given_z = log_bernoulli(x, x_mu, eps=1e-7).sum(axis=1)
    return -kl_term + log_px_given_z, kl_term, log_px_given_z


def gmm_gaussian_lower_bound(z_mu, z_log_var, pz_mu,  pz_log_var, pz_w, x_mu, x_log_var, x):
    """
    Variational Auto-Encoder loss function for gaussian code and output, but with a Gaussian
    Mixture prior.
    """
    kl_term = kl_normal2_gm2(z_mu, z_log_var, pz_mu, pz_log_var, pz_w, eps=1e-6)
    log_px_given_z = log_normal2(x, x_mu, x_log_var, eps=1e-7).sum(axis=1)
    return -kl_term + log_px_given_z, kl_term, log_px_given_z

def gmm_bernoulli_lower_bound(z_mu, z_log_var, pz_mu,  pz_log_var, pz_w, x_mu,  x):
    """
    Variational Auto-Encoder loss function for gaussian code and bernoulli output,
    but with a Gaussian Mixture prior.
    """
    kl_term = kl_normal2_gm2(z_mu, z_log_var, pz_mu, pz_log_var, pz_w, eps=1e-6)
    log_px_given_z = log_bernoulli(x.flatten(), x_mu.flatten(), eps=1e-6).sum(axis=-1)
    return -kl_term + log_px_given_z, kl_term, log_px_given_z

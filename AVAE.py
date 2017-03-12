#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from updates import *

class AVAE(object):
    def __init__(self, in_size, out_size, hidden_size, latent_size,  optimizer = "adadelta"):
        self.prefix = "VAE_"
        self.X = T.matrix("X")
        self.Z = T.matrix("Z")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.optimizer = optimizer

        self.define_layers()
        self.define_train_test_funcs()
        
    def noiser(self, n):
        z = init_normal_weight((n, self.latent_size))
        return floatX(z)
        
    def define_layers(self):
        self.params = []
        
        layer_id = "1"
        self.W_xh = init_weights((self.in_size, self.hidden_size), self.prefix + "W_xh" + layer_id)
        self.b_xh = init_bias(self.hidden_size, self.prefix + "b_xh" + layer_id)

        layer_id = "2"
        self.W_hu = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hu" + layer_id)
        self.b_hu = init_bias(self.latent_size, self.prefix + "b_hu" + layer_id)
        self.W_hsigma = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hsigma" + layer_id)
        self.b_hsigma = init_bias(self.latent_size, self.prefix + "b_hsigma" + layer_id)
 
        self.params += [self.W_xh, self.b_xh, self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma]

        # encoder
        h_enc = T.tanh(T.dot(self.X, self.W_xh) + self.b_xh)
        
        self.mu = T.dot(h_enc, self.W_hu) + self.b_hu
        log_var = T.dot(h_enc, self.W_hsigma) + self.b_hsigma
        self.var = T.exp(log_var)
        self.sigma = T.sqrt(self.var)

        srng = T.shared_randomstreams.RandomStreams(234)
        eps = srng.normal(self.mu.shape)
        self.z = self.mu + self.sigma * eps

        # decoder
        self.G = self.Generator(self.out_size, self.latent_size, self.hidden_size)
        self.params += self.G.params
        self.reconstruct = self.G.generate(self.z)

    class Generator():
        def __init__(self, out_size, latent_size, hidden_size):
            prefix = "gen_"
            self.out_size = out_size
            self.latent_size = latent_size;
            self.hidden_size = hidden_size
            self.Wg_zh = init_weights((self.latent_size, self.hidden_size), prefix + "Wg_zh")
            self.bg_zh = init_bias(self.hidden_size, prefix + "bg_zh")
            self.Wg_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wg_hy")
            self.bg_hy = init_bias(self.out_size, prefix + "bg_hy")
            self.params = [self.Wg_zh, self.bg_zh, self.Wg_hy, self.bg_hy]
        
        def generate(self, z):
            h = T.tanh(T.dot(z, self.Wg_zh) + self.bg_zh)
            y = T.nnet.sigmoid(T.dot(h, self.Wg_hy) + self.bg_hy)
            return y

    class Discriminator():
        def __init__(self, in_size,  hidden_size):
            prefix = "dis_"
            self.in_size = in_size
            self.out_size = 1
            self.hidden_size = hidden_size
            self.Wd_xh = init_weights((self.in_size, self.hidden_size), prefix + "Wd_xh")
            self.bd_xh = init_bias(self.hidden_size, prefix + "bd_xh")
            #self.Wd_hh1 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_hh1")
            #self.bd_hh1 = init_bias(self.hidden_size, prefix + "bd_hh1")
            #self.Wd_hh2 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_hh12")
            #self.bd_hh2 = init_bias(self.hidden_size, prefix + "bd_hh2")
            self.Wd_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wd_hy")
            self.bd_hy = init_bias(self.out_size, prefix + "bd_hy")
            self.params = [self.Wd_xh, self.bd_xh,  self.Wd_hy, self.bd_hy]

        def discriminate(self, x):
            h0 = T.tanh(T.dot(x, self.Wd_xh) + self.bd_xh)
            #h1 = T.tanh(T.dot(h0, self.Wd_hh1) + self.bd_hh1)
            #h2 = T.tanh(T.dot(h1, self.Wd_hh2) + self.bd_hh2)
            y = T.nnet.sigmoid(T.dot(h0, self.Wd_hy) + self.bd_hy)
            return y

    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)

    def define_train_test_funcs(self):
        vlbd = -T.mean((self.kld(self.mu, self.var) + self.multivariate_bernoulli(self.reconstruct, self.X))) 
        
        self.D = self.Discriminator(self.in_size,  self.hidden_size)
        self.params_dis = self.D.params

        g = self.G.generate(self.Z)
        d0 = self.D.discriminate(g)
        d1 = self.D.discriminate(self.reconstruct)
        loss_g = T.mean(-T.log(d0) - T.log(d1))
        gparams = []
        for param in self.params:
            gparam = T.grad(vlbd + loss_g, param)
            gparams.append(gparam)

        
        d2 = self.D.discriminate(self.X)
        loss_d = T.mean(-T.log(d2) - T.log(1 - d0) - T.log(1 - d1))
        gparams_d = []
        for param in self.params_dis:
            gparam = T.grad(loss_d, param)
            gparams_d.append(gparam)

        
        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params + self.params_dis, gparams + gparams_d, lr)
        
        self.train_d = theano.function(inputs = [self.X, self.Z, lr], outputs = [vlbd, loss_d, loss_g], updates = updates)
        self.validate = theano.function(inputs = [self.X], outputs = [vlbd, self.reconstruct])
        self.project = theano.function(inputs = [self.X], outputs = self.mu)
        self.generate = theano.function(inputs = [self.z], outputs = self.reconstruct)
  

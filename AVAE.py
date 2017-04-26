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
        #z = init_uniform_weight((n, self.latent_size))
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
        h_enc = T.nnet.relu(T.dot(self.X, self.W_xh) + self.b_xh)
        
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
            h = T.nnet.relu(T.dot(z, self.Wg_zh) + self.bg_zh, 0.01)
            y = T.nnet.sigmoid(T.dot(h, self.Wg_hy) + self.bg_hy)
            return y

    class DiscriminatorX():
        def __init__(self, in_size,  hidden_size):
            prefix = "dis_x_"
            self.in_size = in_size
            self.out_size = 1
            self.hidden_size = hidden_size
            self.Wd_xh = init_weights((self.in_size, self.hidden_size), prefix + "Wd_xh")
            self.bd_xh = init_bias(self.hidden_size, prefix + "bd_xh")
            #self.Wd_xh2 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_xh2")
            #self.bd_xh2 = init_bias(self.hidden_size, prefix + "bd_xh2")
            self.Wd_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wd_hy")
            self.bd_hy = init_bias(self.out_size, prefix + "bd_hy")
            self.params = [self.Wd_xh, self.bd_xh,  self.Wd_hy, self.bd_hy]

        def discriminate(self, x):
            h0 = T.nnet.relu(T.dot(x, self.Wd_xh) + self.bd_xh, 0.01)
            #h1 = T.nnet.relu(T.dot(h0, self.Wd_xh2) + self.bd_xh2, 0.01)
            #y = T.nnet.sigmoid(T.dot(h0, self.Wd_hy) + self.bd_hy)
            y = T.dot(h0, self.Wd_hy) + self.bd_hy
            return y

    class DiscriminatorZ():
        def __init__(self, in_size,  hidden_size):
            prefix = "dis_z_"
            self.in_size = in_size
            self.out_size = 1
            self.hidden_size = hidden_size
            self.Wd_xh = init_weights((self.in_size, self.hidden_size), prefix + "Wd_xh")
            self.bd_xh = init_bias(self.hidden_size, prefix + "bd_xh")
            self.Wd_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wd_hy")
            self.bd_hy = init_bias(self.out_size, prefix + "bd_hy")
            self.params = [self.Wd_xh, self.bd_xh,  self.Wd_hy, self.bd_hy]

        def discriminate(self, z):
            h0 = T.nnet.relu(T.dot(z, self.Wd_xh) + self.bd_xh, 0.01)
            #y = T.nnet.sigmoid(T.dot(h0, self.Wd_hy) + self.bd_hy)
            y = T.dot(h0, self.Wd_hy) + self.bd_hy
            return y

    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)

    def define_train_test_funcs(self):
        vlbd = -T.mean((self.kld(self.mu, self.var) + self.multivariate_bernoulli(self.reconstruct, self.X))) 
        gparams_vae = []
        for param in self.params:
            gparam = T.grad(vlbd, param)
            gparams_vae.append(gparam)


        self.Dx = self.DiscriminatorX(self.in_size,  self.hidden_size)
        self.params_dis = self.Dx.params
        self.Dz = self.DiscriminatorZ(self.latent_size,  self.hidden_size)
        self.params_dis += self.Dz.params

        # encoder
        d0 = self.Dz.discriminate(self.z) # real
        d1 = self.Dz.discriminate(self.Z) # fake

        # decoder
        g = self.G.generate(self.Z)
        d2 = self.Dx.discriminate(g) # fake
        d3 = self.Dx.discriminate(self.reconstruct) # 0.8 * fake ?
        d4 = self.Dx.discriminate(self.X) # real

        #loss_d = T.mean(-T.log(d0) - T.log(1 - d1) - T.log(d4) - T.log(1 - d2) - T.log(1 - d3)) 
        loss_d = -T.mean(d0) + T.mean(d1) - T.mean(d4) + T.mean(d2) + T.mean(d3)
        gparams_d = []
        for param in self.params_dis:
            gparam = T.grad(loss_d, param)
            gparams_d.append(gparam)

        #loss_g = T.mean(- T.log(d2) - T.log(d3))
        loss_g = - T.mean(d2) - T.mean(d3)
        gparams = []
        for param in self.params:
            gparam = T.grad(vlbd + loss_g, param)
            gparams.append(gparam)
                
        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates_d = optimizer(self.params_dis, gparams_d, lr)
        clip_updates_d = []
        for p, v in updates_d:
            clip_updates_d.append((p, T.clip(v, -0.01, 0.01)))
        updates_d = clip_updates_d
        updates_g = optimizer(self.params, gparams, lr)
        updates_vae = optimizer(self.params, gparams_vae, lr)

        self.train_vae = theano.function(inputs = [self.X, lr], outputs = vlbd, updates = updates_vae)
        self.train_d = theano.function(inputs = [self.X, self.Z, lr], outputs = loss_d, updates = updates_d)
        self.train_g = theano.function(inputs = [self.X, self.Z, lr], outputs = [vlbd,  loss_g], updates = updates_g)
        
        self.validate = theano.function(inputs = [self.X], outputs = [vlbd, self.reconstruct])
        self.project = theano.function(inputs = [self.X], outputs = self.mu)
        self.generate = theano.function(inputs = [self.z], outputs = self.reconstruct)
  

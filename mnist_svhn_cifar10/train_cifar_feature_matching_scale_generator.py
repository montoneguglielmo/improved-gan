import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cifar10_data


## This code is a modification of Goodfellow code for GAN
# The generator receive an image in input and is asked to create an image to fake the discriminator. In order to fake the discriminator,
# the distance between the activation of the discriminator when seeing a real image I and the activation when seeing the output of the generator when the generator has as input I
# is used as error signal for the generator.
# The code trains the discriminator and the generator on CIFAR, it is possible to change the value of the parameter "scale" to rescale the size of cifar up to 4


# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=400)
parser.add_argument('--batch_size', default=20)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/guglielmo/dataset/cifar-10-python')
args = parser.parse_args()
print(args)

from scipy import interpolate
def rescale(img, scale=1):
    scale_fact = scale
    if scale>1:
        x     = np.arange(img.shape[1])
        y     = np.arange(img.shape[2])
        x_new = np.linspace(0, img.shape[1]-1,  img.shape[1] * scale_fact)
        y_new = np.linspace(0, img.shape[2]-1,  img.shape[2] * scale_fact)

        img_out = np.zeros((img.shape[0], int(img.shape[1] * scale_fact), int(img.shape[2] * scale_fact)))
        for cnt in range(3):
            f     = interpolate.interp2d(x,y,img[cnt], kind='linear')
            img_out[cnt] = f(x_new, y_new)
    else:
        img_out = img        
    return np.asarray(img_out, dtype=np.float32)


def tensorRescale(tensor, scale=1):
    tensorScale = np.zeros((tensor.shape[0], 3, int(32*scale), int(32*scale)), dtype=np.float32)
    for cnt in range(tensor.shape[0]):
        tensorScale[cnt] = rescale(tensor[cnt], scale)
    return tensorScale



scale = 2
# fixed random seeds
rng_data   = np.random.RandomState(args.seed_data)
rng        = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx_unl     = trainx.copy()
trainx_unl2    = trainx.copy()
testx, testy   = cifar10_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# specify generative model
x_lab  = T.tensor4()
gen_layers = [ll.InputLayer(shape=(None, 3, 32*scale, 32*scale))]
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 64, (5,5), stride=2, pad=2, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 128, (5,5), stride=2, pad=2, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(gen_layers[-1], 256, (5,5), stride=2, pad=2, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8*scale,8*scale), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4*scale -> 8*scale
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16*scale,16*scale), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8*scale -> 16*scale
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32*scale,32*scale), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 32*scale -> 64*scale
gen_dat = ll.get_output(gen_layers[-1], x_lab)

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32*scale, 32*scale))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_lab  = T.tensor4()
x_unl  = T.tensor4()
temp   = ll.get_output(gen_layers[-1], x_lab, deterministic=False, init=True)
temp   = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]


gen_dat = ll.get_output(gen_layers[-1], x_unl)
#outfun = th.function(inputs=[x_unl], outputs=gen_dat)

output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

l_lab = output_before_softmax_lab[T.arange(args.batch_size),labels]
l_unl = nn.log_sum_exp(output_before_softmax_unl)
l_gen = nn.log_sum_exp(output_before_softmax_gen)
loss_lab = -T.mean(l_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))

train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg   = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens  = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param       = th.function(inputs=[x_lab], outputs=None, updates=init_updates) # data based initialization
train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
samplefun  = th.function(inputs=[x_unl], outputs=gen_dat)

# Theano functions for training the gen net
x_unl2 = T.tensor4()
output_unl = ll.get_output(disc_layers[-2], x_unl2, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)
m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen   = T.mean(abs(output_unl - output_gen))#T.mean(abs(m1-m2)) #+ 0.00001 * T.mean(abs(x_unl-gen_dat))# feature matching loss plus RECONSTRUCTION ERROR
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen   = th.function(inputs=[x_unl, x_unl2, lr], outputs=None, updates=gen_param_updates)

# # select labeled data
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)



# //////////// perform training //////////////
for epoch in range(1200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl  = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    
    if epoch==0:
        trainXscale = tensorRescale(trainx[:500], scale)
        init_param(trainXscale) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    for t in range(nr_batches_train):
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size

        trainXscale    = tensorRescale(trainx[ran_from:ran_to], scale)
        trainXunlScale = tensorRescale(trainx_unl[ran_from:ran_to], scale)

        ll, lu, te = train_batch_disc(trainXscale, trainy[ran_from:ran_to], trainXunlScale, lr)
        
        loss_lab += ll
        loss_unl += lu
        train_err += te

        trainXunlScale2 = tensorRescale(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size], scale)
        train_batch_gen(trainXunlScale, trainXunlScale,lr)
        

    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        testXscale = tensorRescale(testx[t*args.batch_size:(t+1)*args.batch_size], scale)
        test_err  += test_batch(testXscale, testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test

    # report
    print("Iteration %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f" % (epoch, time.time()-begin, loss_lab, loss_unl, train_err, test_err))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun(testXscale[:args.batch_size])
    img_bhwc = np.transpose(sample_x[:16,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig("cifar_sample_generated_feature_match" + str(epoch) + ".png")

    if epoch == 0:
        sample_x = testXscale[:args.batch_size]
        img_bhwc = np.transpose(sample_x[:16,], (0, 2, 3, 1))
        img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples')
        plotting.plt.savefig("cifar_sample_original_feature_match.png")


    
    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])

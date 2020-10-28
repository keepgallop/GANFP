import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from source.miscs.random_samples import sample_continuous, sample_categorical

#----------------------------------------------------------------------------
# Func to extract images' fingerprint features
import tensorflow as tf

extractor_dir   = "../Extractor_models"

with tf.variable_scope("Extractor_weights", reuse=tf.AUTO_REUSE) as scope:
    w_conv1 = tf.get_variable(name="weights1", shape=[3,3,3,3], dtype=tf.float32, trainable=False)
    b_conv1 = tf.get_variable(name="biases1", shape=[3], dtype=tf.float32, trainable=False)

    w_conv2 = tf.get_variable(name="weights2", shape=[3,3,3,8], dtype=tf.float32, trainable=False)
    b_conv2 = tf.get_variable(name="biases2", shape=[8], dtype=tf.float32, trainable=False)

    w_conv3 = tf.get_variable(name='weights3', shape=[3,3,8,16], dtype=tf.float32, trainable=False)
    b_conv3 = tf.get_variable(name='biases3', shape=[16], dtype=tf.float32, trainable=False)

    w_conv4 = tf.get_variable(name='weights4', shape=[3,3,16,32], dtype=tf.float32, trainable=False)
    b_conv4 = tf.get_variable(name='biases4', shape=[32], dtype=tf.float32, trainable=False)

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(extractor_dir)
    if ckpt:
        print("[*]Loading checkpoints from...", ckpt)
        saver = tf.train.Saver(
            var_list={
                'Extract_conv1/weights':w_conv1,'Extract_conv1/biases':b_conv1,
                'Extract_conv2/weights':w_conv2,'Extract_conv2/biases':b_conv2,
                'Extract_conv3/weights':w_conv3,'Extract_conv3/biases':b_conv3,
                'Extract_conv4/weights':w_conv4,'Extract_conv4/biases':b_conv4
            }
        )
        saver.restore(tf.get_default_session(), ckpt)
        print("[*]Loading successfully...")

def feature_extractor(image_batch):
    # image.shape (32, 128, 128, 3)
    #images = tf.transpose(images, perm=[0,3,1,2])
    def dct2d(img_batch):
        with tf.variable_scope("DCT_Extractor", reuse=tf.AUTO_REUSE):
            images_dct = []
            for img_idx in range(img_batch.shape[0]):
                img = img_batch[img_idx]
                layers_dct = []

                for layer_idx in range(img.shape[0]):
                    layer = img[layer_idx]

                    layer_dct = tf.signal.dct(layer, norm='ortho')
                    layer_dct = tf.signal.dct(tf.transpose(layer_dct), norm='ortho')
                    
                    layers_dct.append(layer_dct)
                    
                layers_dct = tf.convert_to_tensor(layers_dct)
                images_dct.append(layers_dct)
                
            images_dct = tf.convert_to_tensor(images_dct)
        return images_dct

    def fingerprint2d(img_batch):
        def conv2d(img, w):
            return tf.nn.conv2d(img, w, strides=[1,1,1,1], padding="SAME", data_format="NCHW")
        def max_pool(img, name):
            return tf.nn.max_pool2d(img, ksize=[1,1,3,3], strides=[1,1,2,2], padding="SAME", name=name, data_format="NCHW")
        with tf.variable_scope("Fingerprint_extractor", reuse=tf.AUTO_REUSE):
            h_conv1 = tf.nn.leaky_relu(tf.nn.bias_add(conv2d(img_batch, w_conv1), b_conv1, data_format="NCHW", name=scope.name))
            h_conv2 = tf.nn.leaky_relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2), b_conv2, data_format="NCHW", name=scope.name))
            pool1 = max_pool(h_conv2, 'pooling1')
            norm1 = tf.nn.leaky_relu(pool1, name='norm1')

            h_conv3 = tf.nn.leaky_relu(tf.nn.bias_add(conv2d(norm1, w_conv3), b_conv3, data_format="NCHW", name=scope.name))
            pool2 = max_pool(h_conv3, 'pooling2')
            norm2 = tf.nn.leaky_relu(pool2, name='norm2')
            
            h_conv4 = tf.nn.leaky_relu(tf.nn.bias_add(conv2d(norm2, w_conv4), b_conv4, data_format="NCHW", name=scope.name))
            pool3 = max_pool(h_conv4, 'pooling3')
            norm3 = tf.nn.leaky_relu(pool3, name='norm3')        
            
            fin_feature = tf.layers.flatten(norm3)
        return fin_feature
    images_dct = dct2d(image_batch)
    images_fin = fingerprint2d(images_dct)

    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    with tf.get_default_session() as sess:
        sess.run(init)

        fin_out = sess.run(images_fin)
    return fin_out
#----------------------------------------------------------------------------

# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        super(Updater, self).__init__(*args, **kwargs)

    def _generete_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        gen = self.models['gen']
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []
        y = []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        for i in range(self.n_dis):
            if i == 0:
                x_fake, y_fake = self._generete_samples()
                dis_fake = dis(x_fake, y=y_fake)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            x_real, y_real = self.get_batch(xp)
            batchsize = len(x_real)
            dis_real = dis(x_real, y=y_real)
            x_fake, y_fake = self._generete_samples(n_gen_samples=batchsize)
            dis_fake = dis(x_fake, y=y_fake)
            x_fake.unchain_backward()

            print("#X.shape: ", x_real.shape, x_fake.shape)
            
            # Add anonymize loss
            fin_weight = 1.0
            real_features_out = feature_extractor(x_real)
            fake_features_out = feature_extractor(x_fake)
            print("#real_features_out.shape: ", real_features_out.shape)
            print("#fake_features_out.shape: ", fake_features_out.shape)
            fin_loss = F.mean(F.absolute(real_features_out[1:,:,:,:] - fake_features_out[1:,:,:,:]))
            print("#fin_loss.shape: ", fin_loss.shape)
            fin_loss *= fin_weight

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            print("#loss_dis.shape: ", loss_dis.shape)
            loss_dis += fin_loss
            
            print("#loss.shape: ", loss_gen.shape, loss_dis.shape)

            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})

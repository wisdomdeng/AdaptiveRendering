from __future__ import division
from __future__ import absolute_import

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

import argparse
from datetime import datetime
import math
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin

import vgg
from create_database import *
from create_database_svm import *

import functools

FLAGS = None

STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2')
CONTENT_LAYER = 'relu4_2'

class pose2image(object):
    def __init__(self, sess, image_size=128,
                 batch_size=1, sample_size=1, output_size=128,
                 gf_dim=32, df_dim=32, L1_lambda=100,
                 input_pose_dim=3, input_style_dim=3, output_image_dim=3, dataset_name='facades',
                 content_weight=1e-4, style_weight=1e-14,
                 checkpoint_dir=None, sample_dir=None,
                 vgg_path='/local-scratch2/mzhai/cvpr18/dataset/data/pre_trained/beta16/imagenet-vgg-verydeep-19.mat'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_style_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.vgg_path = vgg_path

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_style_dim = input_style_dim
        self.input_pose_dim = input_pose_dim
        self.output_image_dim = output_image_dim

        self.L1_lambda = L1_lambda

        self.style_weight = style_weight
        self.content_weight = content_weight

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_pose_dim + self.input_style_dim + self.output_image_dim],
                                        name='pose_and_style')

        self.style = self.real_data[:, :, :, :self.input_style_dim]
        self.pose = self.real_data[:, :, :, self.input_style_dim:self.input_style_dim + self.input_pose_dim]
        self.target = self.real_data[:, :, :, self.input_style_dim + self.input_pose_dim:self.input_style_dim + self.input_pose_dim + self.output_image_dim]

        #print self.input_style_dim, self.input_pose_dim, self.output_image_dim

        self.fake_target = self.generator(self.pose, self.style)

        self.real_target_with_style = tf.concat([self.pose, self.target], 3)
        self.fake_target_with_style = tf.concat([self.pose, self.fake_target], 3)
        self.D, self.D_logits = self.discriminator(self.real_target_with_style, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_target_with_style, reuse=True)


        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_target_sum = tf.summary.image("fake_target", self.fake_target)


        # style loss
        style_pre = vgg.preprocess((self.style+1.0)*127.5)
        net_style = vgg.net(self.vgg_path, style_pre)

        fake_target_pre = vgg.preprocess((self.fake_target+1.0)*127.5)
        net_fake_target = vgg.net(self.vgg_path, fake_target_pre)

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net_fake_target[style_layer]
            # batch_size, h, w, channel
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams_fake_target = tf.matmul(feats_T, feats)

            layer = net_style[style_layer]
            # batch_size, h, w, channel
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams_style = tf.matmul(feats_T, feats)

            style_losses.append(2 * tf.nn.l2_loss(grams_fake_target - grams_style))

        self.style_loss = self.style_weight * functools.reduce(tf.add, style_losses) /(int(grams_style.shape[0])*int(grams_style.shape[1])*int(grams_style.shape[2]))


        # content loss
        fake_target_pre = vgg.preprocess((self.fake_target+1.0)*127.5)
        net_fake_target = vgg.net(self.vgg_path, fake_target_pre)

        target_pre = vgg.preprocess((self.target+1.0)*127.5)
        net_target = vgg.net(self.vgg_path, target_pre)

        content_losses = []
        feats_fake_target = net_fake_target[CONTENT_LAYER]
        feats_target = net_target[CONTENT_LAYER]

        self.content_loss = self.content_weight * (2 * tf.nn.l2_loss(feats_fake_target - feats_target) / (int(feats_target.shape[0])*int(feats_target.shape[1])*int(feats_target.shape[2])*int(feats_target.shape[3])))


        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.target - self.fake_target)) + self.content_loss + self.style_loss
        #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) + self.content_loss + self.style_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, batch_images, sample_dir, epoch, idx):
        sample_images = batch_images
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_target, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    '{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pose2image"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum, self.fake_target_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load_old(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # data = create_database('train', 1)
        # batch_idxs = min(len(data), args.train_size) // self.batch_size
        data_test = create_database_svm('test', 1)
        batch_idxs_test = len(data_test) // self.batch_size

        for epoch in xrange(args.epoch):

            data = create_database('train', 1)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim three times to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # # Run g_optim four times to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str = self.sess.run([g_optim, self.g_sum],
                #                                feed_dict={ self.real_data: batch_images })
                # self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                errContent = self.content_loss.eval({self.real_data: batch_images})
                errStyle = self.style_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f, content_loss: %.4f, style_loss: %.4f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG, errContent, errStyle))

                if np.mod(counter, 100) == 1:
                    self.sample_model(batch_images, args.sample_dir, epoch, idx)

                if np.mod(counter, 1000) == 2:
                    self.save_old(args.checkpoint_dir, counter)
            
            if np.mod(epoch, 10) == 0:
                TEST_DIR = args.test_dir + '/debug_epoch_' + str(epoch+1)
                if not os.path.exists(TEST_DIR):
                    os.makedirs(TEST_DIR)
                for idx in xrange(0, batch_idxs_test):
                    batch_files_test = data_test[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_test = [load_data(batch_file_test) for batch_file_test in batch_files_test]
                    if (self.is_grayscale):
                        batch_images_test = np.array(batch_test).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images_test = np.array(batch_test).astype(np.float32)

                    samples_test = self.sess.run(
                        self.fake_target,
                        feed_dict={self.real_data: batch_images_test}
                    )
                    print samples_test.shape
                    person_idx = idx + 1
                    save_images(samples_test, [self.batch_size, 1],
                                '{}/test_{:d}.png'.format(TEST_DIR, person_idx))

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 128 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (32 x 32 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (16 x 16 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (8 x 8 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, pose, style, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

            # pose is (128 x 128 x input_pose_dim)
            e1 = conv2d(pose, self.gf_dim, name='g_e1_conv')
            # e1 is (64 x 64 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (32 x 32 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (16 x 16 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e4), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), 64, name='g_e7_conv'))
            # e7 is (2 x 2 x 10)

            input_filt_dim = lrelu(e7).get_shape()[-1]
            output_filt_dim = 64


            ## Compute the adaptive filter given reference image
            ## Make sure ada_filt size is [5,5,lrelu(e7).get_shape()[-1],self.gf_dim*8]
            ada_filt = adaptive_filter(style, input_filt_dim, output_filt_dim)
            ## Last conv layer, use adaptive convolution
            conv = tf.nn.conv2d(lrelu(e7), ada_filt, strides=[1, 2, 2, 1], padding='SAME', name='g_e8_conv')
            biases = tf.get_variable('biases', [output_filt_dim], initializer=tf.constant_initializer(0.0))
            e8 = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            #e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)


            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (8 x 8 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (16 x 16 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (32 x 32 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (64 x 64 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_image_dim], name='g_d8', with_w=True)
            # d8 is (128 x 128 x output_c_dim)

            return tf.nn.tanh(self.d8)


    # def save(self, checkpoint_dir, step):
    #     model_name = "pose2image.model"
    #     model_dir = "%s_%s_%d" % (self.dataset_name, "Epoch", step)
    #     checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)

    #     self.saver.save(self.sess,
    #                     os.path.join(checkpoint_dir, model_name),
    #                     global_step=step)

    # def load(self, checkpoint_dir, step = 1000):
    #     print(" [*] Reading checkpoint...")

    #     model_dir = "%s_%s_%d" % (self.dataset_name, "Epoch", step)
    #     checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         return True
    #     else:
    #         return False

    def save_old(self, checkpoint_dir, step):
        model_name = "pose2image.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_old(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):
        """Test pose2image"""
        print("Init variables...")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Variables initialized")
        print("Creating Test Data...")
        data = create_database_svm('test', 1)
        print("Data created...")
        batch_idxs = min(len(data), args.train_size) // self.batch_size
        start_time = time.time()
        if self.load_old(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [load_data(batch_file) for batch_file in batch_files]
            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)

            samples = self.sess.run(
                self.fake_target,
                feed_dict={self.real_data: batch_images}
            )
            print samples.shape
            person_idx = idx + 1
            save_images(samples, [self.batch_size, 1],
                        '{}/test_{:d}.png'.format(args.test_dir, person_idx))

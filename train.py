import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname("dev"))))))
sys.path.append('/home/dkkim/workspace/dev')

import tensorflow as tf
from glob import glob
import random, json, os, numpy as np
from model import convmodel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import data_load, original_load, npy_load, tower_loss, average_gradients, send_email

flags = tf.app.flags
flags.DEFINE_integer("num_gpu", 2, help="Number of GPUs")
flags.DEFINE_integer("batch_size", 24, help="batch_size")
flags.DEFINE_float("lr_init", 0.008, help="lr init")
flags.DEFINE_string("save_path", "/data/dk/models/SEdense_test", help="Directory name to save the weights and records")
flags.DEFINE_string("data_path", "/data/dk/type_crops", help="Directory to load data")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "0,1", help="GPU number")

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICES



class Model():
    def __init__(self):

        if not os.path.exists(FLAGS.save_path + '/weights'): os.makedirs(FLAGS.save_path + 'weights')
        if not os.path.exists(FLAGS.save_path + '/results'): os.makedirs(FLAGS.save_path + 'results')

        self.gpu_num = list(range(FLAGS.num_gpu))
        self.batch, self.hbatch = FLAGS.batch_size, FLAGS.batch_size / 3
        self.res_check_step = 50
        self.model_nums, self.train_accs, self.steps, \
        self.val_snaccs, self.val_psnaccs, self.val_nsnaccs, self.val_snlosses, self.val_psnlosses, self.val_nsnlosses = [], [], [], [], [], [], [], [], []


    def dataset(self):

        self.t_sn = data_load('%s/train/solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/train2/solid/*/*/*/*/*/')
        self.t_psn = data_load('%s/train/part-solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/train2/part-solid/*/*/*/*/*/')
        self.t_nsn = data_load('%s/train/non-solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/train2/non-solid/*/*/*/*/*/')
        self.v_sn = original_load('%s/valid/solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/valid2/solid/*/*/*/*/*/')
        self.v_psn = original_load('%s/valid/part-solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/valid2/part-solid/*/*/*/*/*/')
        self.v_nsn = original_load('%s/valid/non-solid/*/*/*/*/*/'%(FLAGS.data_path), '/data/dk/type_crops/valid2/non-solid/*/*/*/*/*/')

        self.t_label = np.append(2*np.ones([self.hbatch,1]),np.append(np.ones([self.hbatch,1]), np.zeros([self.hbatch,1]), axis=0), axis=0)
        self.snv_label, self.psnv_label, self.nsnv_label = 2*np.ones([self.batch,1]), np.ones([self.batch,1]), np.zeros([self.batch,1])

    def tf_model(self,is_finetune=False,is_loadmodel=True):

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            lr_init = FLAGS.lr_init
            self.lr = tf.train.exponential_decay(lr_init, self.global_step, 5000, 0.5, staircase=True)

            # Create an optimizer that performs gradient descent.
            opt = tf.train.RMSPropOptimizer(self.lr, decay=0.9, epsilon=0.1)

            tower_grads = []
            self.towers = {}

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu in self.gpu_num:
                    with tf.device('/gpu:%d' % gpu):
                        with tf.name_scope('%s_%d' % ('tower', gpu)) as scope:
                            num = str(gpu)
                            self.towers['img' + num], self.towers['label' + num], self.towers['loss' + num], \
                            self.towers['tacc' + num], self.towers['is_training' + num], self.towers['prob' + num], total_loss \
                                = tower_loss(scope)
                            tf.get_variable_scope().reuse_variables()
                            batchnorm_updates = tf.get_collection('_update_ops_', scope)
                            if is_finetune:
                                vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                                train_vars = []
                                for i in vars:
                                    if 'fc' in str(i):
                                        train_vars.append(i)
                                    elif 'SE' in str(i):
                                        train_vars.append(i)
                                print train_vars

                                tower_grads.append(opt.compute_gradients(total_loss, var_list=train_vars))
                            else:
                                tower_grads.append(opt.compute_gradients(total_loss))

            grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

            self.sess_config = tf.ConfigProto(allow_soft_placement=True)
            self.sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.sess_config)

            init = tf.global_variables_initializer()
            self.sess.run(init)

            if is_finetune:
                ## load variables list

                vars = tf.global_variables()
                # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

                # train_vars =[]
                reuse_vars = []
                for idx, i in enumerate(vars):
                    if 'fc' in str(i):
                        continue

                    elif 'SE' in str(i):
                        continue

                    else:
                        reuse_vars.append(i)

                reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
                restore_saver = tf.train.Saver(reuse_vars_dict)
                restore_saver.restore(self.sess, FLAGS.save_path + '/weights/model-87')

            self.saver = tf.train.Saver(max_to_keep=1000)

            if is_loadmodel:
                self.saver.restore(self.sess, FLAGS.save_path + '/weights/model-0')

                # lr_init = 0.004
                # self.lr = tf.train.exponential_decay(lr_init, self.global_step, 3000, 0.5, staircase=True)
                # self.sess.run(self.global_step.initializer)

            self.fetches, self.valFetches = [train_op], []
            for gpu in self.gpu_num:
                self.fetches += [self.towers['loss' + str(gpu)], self.towers['tacc' + str(gpu)]]
                self.valFetches += [self.towers['loss' + str(gpu)], self.towers['tacc' + str(gpu)]]

    def run_model(self):

        self.acc_sum, self.loss_sum, self.model_num = 0,0,0
        self.idx = 0

        for idx, i in enumerate(range(50000),1):

            self.idx = idx

            feed_dict = {}
            for gpu in self.gpu_num:
                feed_dict[self.towers['img'+str(gpu)]] = \
                   npy_load(random.sample(self.t_sn, self.hbatch) + random.sample(self.t_psn, self.hbatch)+random.sample(self.t_nsn, self.hbatch))
                feed_dict[self.towers['label'+str(gpu)]] = self.t_label
                feed_dict[self.towers['is_training'+str(gpu)]] = True

            cur_res = self.sess.run(self.fetches, feed_dict=feed_dict)
            cur_res = cur_res[1:]

            for x in cur_res[::2]:
                assert not np.isnan(x), 'Model diverged with loss'

            self.loss_sum += np.mean(cur_res[::2])
            self.acc_sum += np.mean(cur_res[1::2]) * 100

            if self.idx % self.res_check_step == 0:
                if self.idx % (self.res_check_step *10) == 0:
                    self.res_check(is_save=True)
                else:
                    self.res_check(is_save=False)

    def res_check(self, is_save = False):

        self.loss_sum /= self.res_check_step
        self.acc_sum /= self.res_check_step

        print '*' * 20
        print "step: %d, model: %d, acc: %.2f%%, loss: %f, global_step: %d, lr: %f" \
              % (self.idx, self.model_num, round(self.acc_sum, 2), round(self.loss_sum, 6),
                 int(self.sess.run(self.global_step)), round(self.sess.run(self.lr), 8))

        if self.model_num == 0:
            self.saver.save(self.sess, save_path=FLAGS.save_path + '/weights/model', global_step=self.model_num)

        if self.acc_sum >= 99 or is_save:
            nsnloss, nsnacc = self.validation_run(is_type='non-solid')
            print "nsn_val_loss: %f, nsn_val_acc: %.2f%%" % (nsnloss, nsnacc)


            if (nsnacc >= 70 and nsnloss < 0.5) or is_save:
                psnloss, psnacc = self.validation_run(is_type='part-solid')
                print "psn_val_loss: %f, psn_val_acc: %.2f%%" % (psnloss, psnacc)

                if (psnacc >= 90 and psnloss < 0.5 ) or nsnacc >= 87 or is_save:
                    snloss, snacc = self.validation_run(is_type='solid')
                    print "sn_val_loss: %f, sn_val_acc: %.2f%%" % (snloss, snacc)

                    if snacc >= 80 or is_save:
                        self.saver.save(self.sess, save_path=FLAGS.save_path + '/weights/model',
                                        global_step=self.model_num)

                    Subject = "Results"
                    Text = "step: %d, model_num: %d \n" \
                           "\t train_acc: %.2f%%, train_loss: %.5f \n" \
                           "\t val_sn_acc: %.2f%%, val_sn_loss: %.5f \n" \
                           "\t val_psn_acc: %.2f%%, val_psn_loss: %.5f \n"\
                            "\t val_nsn_acc: %.2f%%, val_nsn_loss: %.5f \n"\
                           % (self.idx, self.model_num, self.acc_sum, self.loss_sum,
                              snacc, snloss, psnacc, psnloss, nsnacc, nsnloss)

                    send_email(Subject,Text)

                    self.model_nums.append(self.model_num)
                    self.train_accs.append(round(self.acc_sum,2))
                    self.steps.append(self.idx)
                    self.val_snaccs.append(snacc)
                    self.val_psnaccs.append(psnacc)
                    self.val_nsnaccs.append(nsnacc)
                    self.val_snlosses.append(snloss)
                    self.val_psnlosses.append(psnloss)
                    self.val_nsnlosses.append(nsnloss)

                    save_data = [{'model_num': self.model_nums[q],
                                  'train_acc': self.train_accs[q],
                                  'step': self.steps[q],
                                  'val_sn_acc': self.val_snaccs[q],
                                  'val_psn_acc': self.val_psnaccs[q],
                                  'val_nsn_acc': self.val_nsnaccs[q],
                                  'val_sn_loss': self.val_snlosses[q],
                                  'val_psn_loss': self.val_psnlosses[q],
                                  'val_nsn_loss': self.val_nsnlosses[q]}
                                 for q in range(len(self.steps))]

                    with open(FLAGS.save_path + '/results/record.json', 'wb') as f:
                        json.dump(save_data,f)

                    self.model_num += 1

        self.loss_sum, self.acc_sum = 0., 0.


    def validation_run(self, is_type):

        cnt = 0
        val_loss_sum, val_acc_sum = 0, 0

        if is_type == 'solid':
            cur_img = self.v_sn
            cur_label = self.snv_label
        elif is_type == 'part-solid':
            cur_img = self.v_psn
            cur_label = self.psnv_label
        elif is_type == 'non-solid':
            cur_img = self.v_nsn
            cur_label = self.nsnv_label

        for vstart in range(0, len(cur_img),self.batch * FLAGS.num_gpu):
            if len(cur_img) - vstart <= self.batch *  FLAGS.num_gpu: continue
            cnt+=1

            sys.stdout.write('val_step: {}\r'.format(vstart))

            feed_dict = {}
            for gpu in self.gpu_num:
                feed_dict[self.towers['img' + str(gpu)]] = npy_load(cur_img[vstart + FLAGS.batch_size * gpu:
                                                                   vstart + FLAGS.batch_size * (gpu + 1)])
                feed_dict[self.towers['label' + str(gpu)]] = cur_label
                feed_dict[self.towers['is_training' + str(gpu)]] = False

            cur_res = self.sess.run(self.valFetches, feed_dict=feed_dict)

            val_loss_sum += np.mean(cur_res[::3])
            val_acc_sum += np.mean(cur_res[1::3]) * 100
            # probs = cur_res[2::3]

        val_loss_sum /= cnt
        val_acc_sum /= cnt

        return round(val_loss_sum, 5), round(val_acc_sum, 2)

def main():

    try:
        model = Model()
        model.dataset()
        model.tf_model()
        model.run_model()
    except Exception as e:
        Text = str(e)
        send_email(Subject='Error occured',Text=Text)

if __name__ == '__main__':
    main()

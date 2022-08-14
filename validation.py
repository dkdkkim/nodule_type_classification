import json, os, sys
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
from utils import Database, original_load, npy_load, tower_loss

flags = tf.app.flags
flags.DEFINE_integer("num_gpu", 1, help="Number of GPUs")
flags.DEFINE_integer("batch_size", 4, help="batch_size")
flags.DEFINE_string("save_path", "/data/dk/models/final/type_classification", help="Directory name to save the weights and records")
flags.DEFINE_string("data_path", "/data/dk/type_crops", help="Directory to load data")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "1", help="GPU number")
FLAGS = flags.FLAGS

class type():

    def __init__(self):
        self.batch, self.hbatch = FLAGS.batch_size, FLAGS.batch_size / 3
        self.model_no = '37'
        self.is_save = False
        self.vis = False
        self.res_path = '/data/dk/result/type_test'
        if not os.path.exists(self.res_path): os.mkdir(self.res_path)
        self.gpu_num = list(range(FLAGS.num_gpu))
        self.count_sn, self.count_psn, self.count_nsn = 0, 0, 0
        self.total_sn, self.total_psn, self.total_nsn, self.total_pass = 0, 0, 0, 0
        self.val_acc, self.val_loss = 0, 0

    def tf_model(self):
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
        self.valFetches = []
        for gpu in self.gpu_num:
            self.valFetches += [self.towers['loss' + str(gpu)], self.towers['tacc' + str(gpu)],self.towers['prob'+str(gpu)]]

        saver = tf.train.Saver(max_to_keep=1000)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver.restore(self.sess, FLAGS.save_path + '/model-' + self.model_no)

    def res_check(self, is_type):
        '''
        ex) res_check('part-solid')
        check the model with one type validation set
        :param is_type: type of input validation set
        '''
        if is_type == 'solid':
            v_label = 2 * np.ones([self.batch, 1])
        elif is_type == 'part-solid':
            v_label = np.ones([self.batch, 1])
        elif is_type == 'non-solid':
            v_label = np.zeros([self.batch, 1])

        val_lists = [FLAGS.data_path + '/valid2/' + is_type + '/*/*/*/*/*/',
                             FLAGS.data_path + '/valid/' + is_type + '/*/*/*/*/*/']

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            val_loss, val_acc, count_nsn, count_psn, count_sn = \
                self.validation_detail(val_lists, v_label,is_type=is_type, model_no=self.model_no,
                                       save_path=self.res_path, is_save=self.is_save)

        print "loss: %f, acc: %.2f%%, solid: %d, part-solid: %d, non-solid: %d" % (
        val_loss, val_acc, count_sn, count_psn, count_nsn)

    def res_check_alltype(self):
        '''
        check the model with all types validation set(solid,part-solid,non-solid)
        :return:
        '''

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            type_list = ['solid', 'part-solid', 'non-solid']

            for is_type in type_list:
                print '<%s>' % is_type
                val_lists = ['/data/dk/type_crops/valid2/' + is_type + '/*/*/*/*/*/',
                             '/data/dk/type_crops/valid/' + is_type + '/*/*/*/*/*/']
                if is_type == 'solid':
                    v_label = 2 * np.ones([self.batch, 1])
                elif is_type == 'part-solid':
                    v_label = np.ones([self.batch, 1])
                elif is_type == 'non-solid':
                    v_label = np.zeros([self.batch, 1])

                val_loss, val_acc, count_nsn, count_psn, count_sn = \
                    self.validation_detail(val_lists, v_label, \
                                           is_figure=True, is_type=is_type, model_no=self.model_no,
                                           save_path=self.res_path, is_save=self.is_save)
                self.total_sn += count_sn
                self.total_psn += count_psn
                self.total_nsn += count_nsn
                self.total_pass += int(val_acc * (count_nsn + count_psn + count_sn))

                print "loss: %f, acc: %.2f%%, solid: %d, part-solid: %d, non-solid: %d" % (
                    val_loss, val_acc, count_sn, count_psn, count_nsn)

            print '\ntotal acc =', float(self.total_pass) / float(self.total_sn + self.total_psn + self.total_nsn)

    def res_check_alltype_DB(self):
        '''
        check the model with all types validation set(solid,part-solid,non-solid)
        :return:
        '''

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            type_list = ['solid', 'part-solid', 'non-solid']
            db = Database()

            for is_type in type_list:
                print '<%s>' % is_type
                val_lists = ['/data/dk/type_crops/valid2/' + is_type + '/*/*/*/*/*/',
                             '/data/dk/type_crops/valid/' + is_type + '/*/*/*/*/*/']

                if is_type == 'solid':
                    v_label = 2 * np.ones([self.batch, 1])
                elif is_type == 'part-solid':
                    v_label = np.ones([self.batch, 1])
                elif is_type == 'non-solid':
                    v_label = np.zeros([self.batch, 1])

                val_loss, val_acc, count_nsn, count_psn, count_sn = \
                    self.validation_detail(val_lists, v_label, \
                                           is_figure=True, is_type=is_type, model_no=self.model_no,
                                           save_path=self.res_path, is_save=self.is_save)
                self.total_sn += count_sn
                self.total_psn += count_psn
                self.total_nsn += count_nsn
                self.total_pass += int(val_acc * (count_nsn + count_psn + count_sn))

                print "loss: %f, acc: %.2f%%, solid: %d, part-solid: %d, non-solid: %d" % (
                    val_loss, val_acc, count_sn, count_psn, count_nsn)

            print '\ntotal acc =', float(self.total_pass) / float(self.total_sn + self.total_psn + self.total_nsn)


    def validation_detail(self, npy_path, label, mode='fixed', **kwargs):

        if not kwargs:
            print kwargs

        if mode == 'select':
            db = Database()
            try:
                output = db.select("select typeSplit,noduleType,iskind,patientID,studyDate,studyUID,seriesUID,coordZ,coordY,coordX \
                                      from crops where (typeSplit = 'valid' or typeSplit = 'valid2') and noduleType = '%s' and KVP = 120\
                                      and convolutionKernel = 'C' and mAs < 50 and thickness < 1.5" % kwargs['is_type'])
            except Exception as e:
                print e
            finally:
                db.conn.close()
            npy_path = []
            for row in output:
                path = '/'.join(['/data/dk/type_crops', row[0], row[1], row[2], row[3], str(row[4]), row[5], row[6],
                                 '_'.join([str(row[7]), str(row[8]), str(row[9]), '*'])])
                npy_path.append(path.encode("utf-8"))

            npy = original_load(npy_path)
        elif mode == 'fixed':
            # npy = data_load(npy_path)
            npy = original_load(npy_path)

        col = 0
        if kwargs['is_type'] == 'part-solid':
            col = 1
        elif kwargs['is_type'] == 'solid':
            col = 2

        result_dict = {'pass': [], 'fail': []}

        for vstart in range(0, len(npy) / (FLAGS.batch_size * len(self.gpu_num)) * (FLAGS.batch_size * len(self.gpu_num)),
                            FLAGS.batch_size * len(self.gpu_num)):

            sys.stdout.write('val_step: {}\r'.format(vstart))

            feed_dict = {}
            for gpu in self.gpu_num:
                feed_dict[self.towers['img' + str(gpu)]] = npy_load(npy[vstart + FLAGS.batch_size * gpu:
                                                                   vstart + FLAGS.batch_size * (gpu + 1)])
                feed_dict[self.towers['label' + str(gpu)]] = label
                feed_dict[self.towers['is_training' + str(gpu)]] = False

            cur_res = self.sess.run(self.valFetches, feed_dict=feed_dict)
            if not kwargs:
                self.val_loss += np.mean(cur_res[::2])
                self.val_acc += np.mean(cur_res[1::2])
            else:
                self.val_loss += np.mean(cur_res[::3])
                self.val_acc += np.mean(cur_res[1::3])
                probs = cur_res[2::3]

                # count prob
                for gpu in self.gpu_num:
                    for imgidx, prob in enumerate(probs[gpu]):

                        if prob[0] == prob.max():
                            type_result = 'non-solid'
                            print 'nsn \n',npy[vstart + FLAGS.batch_size * gpu + imgidx]

                            self.count_nsn += 1
                        elif prob[1] == prob.max():
                            type_result = 'part-solid'
                            print 'psn \n',npy[vstart + FLAGS.batch_size * gpu + imgidx]

                            self.count_psn += 1
                        elif prob[2] == prob.max():
                            type_result = 'solid'

                            self.count_sn += 1

                        if prob[col] != prob.max():
                            pf = 'fail'
                        else:
                            pf = 'pass'

                        result_dict[pf].append([npy[vstart + FLAGS.batch_size * gpu + imgidx], round(prob[0], 3),
                                                round(prob[1], 3), round(prob[2], 3), type_result])
        self.val_loss /= (divmod(vstart, FLAGS.batch_size * len(self.gpu_num))[0] + 1)
        self.val_acc /= (divmod(vstart, FLAGS.batch_size * len(self.gpu_num))[0] + 1)

        print 'fail : %d, pass : %d' % (len(result_dict['fail']), len(result_dict['pass']))
        if kwargs['is_save']:
            with open(kwargs['save_path'] + '/result_path_' + kwargs['is_type'] + '_' + kwargs['model_no'] + '.json',
                      'w') as outfile:
                json.dump(result_dict, outfile)

        return round(self.val_loss, 6), round(self.val_acc * 100, 2), self.count_nsn, self.count_psn, self.count_sn


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    model = type()
    model.tf_model()
    model.res_check('solid')
    # model.res_check_alltype()

if __name__ == '__main__':
    main()

import json, os, pymysql,sys
import matplotlib
matplotlib.use("Agg")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glob import glob
from model import convmodel

flags = tf.app.flags
flags.DEFINE_string("save_path", "/data/dk/models/final/type_classification", help="Directory name to save the weights and records")
flags.DEFINE_string("data_path", '/data/dk/json/fat_pos_DB.json', help="JSON file path of data")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "1", help="GPU number")
FLAGS = flags.FLAGS

class type():

    def __init__(self, ):
        self.model_no = '37'

    def tf_model(self):

        model = convmodel()
        self.image, self.is_training, prob, out1, out2 = model.convnet_test()
        self.valFetches = [prob]

        saver = tf.train.Saver(max_to_keep=1000)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver.restore(self.sess, FLAGS.save_path + '/model-' + self.model_no)

    def res_check(self,data_path):

        img = np.load(data_path)
        input = np.expand_dims(img[8:40,...], axis=0)
        input = np.expand_dims(input, axis=4)

        cur_res = self.sess.run(self.valFetches, feed_dict={self.image: input, self.is_training: False})
        prob_out = cur_res[0]

        if prob_out.max() == prob_out[0]:
            result = 'non-solid'
        elif prob_out.max() == prob_out[1]:
            result = 'part-solid'
        elif prob_out.max() == prob_out[2]:
            result = 'solid'

        # fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        # for i in range(16):
        #     ax[i / 4, i % 4].imshow(img[8 + i, ...], cmap='bone', clim=(0., 1.5))
        # plt.show()

        return result

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    model = type()
    model.tf_model()

    with open(FLAGS.data_path,'rb') as file:
        fats = json.load(file)
    
    res = []
    for idx,fat in enumerate(fats):
        print idx
        res.append([fat,model.res_check(fat)])
    
    with open('%s/fat_pos_DB_withtype.json'%(FLAGS.save_path),'w') as outfile:
        json.dump(res,outfile)


if __name__ == '__main__':
    main()

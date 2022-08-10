import json, os, pymysql,sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import scipy.stats as st
from utils import Database, original_load, npy_load
import math,csv
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("num_gpu", 1, help="Number of GPUs")
flags.DEFINE_integer("batch_size", 2, help="batch_size")
flags.DEFINE_string("save_path", "/data/dk/models/final/type_classification", help="Directory name to save the weights and records")
flags.DEFINE_string("CUDA_VISIBLE_DEVICES", "1", help="GPU number")
flags.DEFINE_string("mode", "CAM", help="ananysis mode")
FLAGS = flags.FLAGS

class analysis_type():
    def __init__(self):
        self.db = Database()
        self.model_no = '37'

    def load_result(self,json_path):

        '''
        1. load json file from res-check
        2. make dictionary for analysis
        :param json_path: directory where result json is
        :return:

        ex) load_result('/data/dk/result/SEdense_r16_ver2')
        '''

        save_path = json_path
        type_list = ['solid', 'part-solid', 'non-solid']
        result = {'pass': [], 'fail': []}

        for is_type in type_list:
            with open(save_path + '/result_path_' + is_type + '_37.json', 'rb') as infile:
                res = json.load(infile)

            result['pass'] += res['pass']
            result['fail'] += res['fail']

        print 'pass : %d, fail : %d' % (len(result['pass']), len(result['fail']))

        pf_list = ['pass','fail']

        detail = {'pass':{},'fail':{}}

        for pf in pf_list:
            detail[pf] = {'x': [], 'y': [], 'z': [], 'rotation': [], 'transpose': [], 'interval': [], 'thickness': [],
                       'iskind': [], 'mA': [], 'KVP': [], 'mAs': [], 'convolutionKernel': [],'noduleType':[],'diameter':[]}

            for didx, data in enumerate(result[pf]):
                if didx % 100 == 0: print pf, didx

                filename = data[0].split('/')[-1]
                seriesUID = data[0].split('/')[-2]
                noduleType = data[0].split('/')[5]

                detail[pf]['z'].append(int(filename.split('_')[0].encode("ascii")))
                detail[pf]['y'].append(int(filename.split('_')[1].encode("ascii")))
                detail[pf]['x'].append(int(filename.split('_')[2].encode("ascii")))
                detail[pf]['rotation'].append(int(filename.split('_')[-2].encode("ascii")))
                detail[pf]['transpose'].append(int(filename.split('_')[-1][0:3].encode("ascii")))
                detail[pf]['noduleType'].append(noduleType.encode("ascii"))

                try:
                    sql = "select intervals,thickness,iskind,mA,kvp,mAs,convolutionKernel,averageDiameter from crops where seriesUID = '%s' \
                                        and coordZ = %s and coordY = %s and coordX = %s" % (
                    seriesUID, int(filename.split('_')[0].encode("ascii")), \
                    int(filename.split('_')[1].encode("ascii")), int(filename.split('_')[2].encode("ascii")))
                    info = self.db.select(sql)
                except Exception as e:
                    print 'select error :',e
                    print seriesUID,filename

                if len(info) == 0:
                    print 'no data, seriesUID :', seriesUID
                    continue
                keys = ['interval','thickness','iskind','mA','KVP','mAs','convolutionKernel','diameter']
                for kidx,key in enumerate(keys):
                    detail[pf][key].append(info[0][kidx])

        return detail

    def plot_histogram(self,key_list,res_dict,is_save=False,save_path=None):
        '''

        :param key_list: list type. 'x', 'y', 'z', 'rotation', 'transpose', 'interval', 'thickness', 'iskind', 'mA', 'KVP', 'mAs',
        #                 'convolutionKernel', 'diameter', 'noduleType'
        :param res_dict: result from load_result
        :param is_save: whether saving the graph or not
        :param save_path: save path for graphs
        :return: none
        '''

        # key_list = ['x', 'y', 'z', 'rotation', 'transpose', 'interval', 'thickness', 'iskind', 'mA', 'KVP', 'mAs',
        #                 'convolutionKernel']
        for idxk, keyword in enumerate(key_list):
            n_step = 10

            if keyword == 'iskind' or keyword == 'convolutionKernel' or keyword == 'noduleType':
                p_list = list(set(res_dict['pass'][keyword]))
                f_list = list(set(res_dict['fail'][keyword]))
                x_list = list(set(p_list + f_list))

                p_dict, f_dict = {}, {}
                for idx, x in enumerate(x_list):
                    p_dict[x] = 0
                    f_dict[x] = 0
                for i in res_dict['pass'][keyword]:
                    p_dict[i] += 1
                for i in res_dict['fail'][keyword]:
                    f_dict[i] += 1
                w = 0.2

                py = [p_dict[x] for x in x_list]
                fy = [f_dict[x] for x in x_list]

                if len(py) < 2:
                    py = [py[0], 0]
                    fy = [fy[0], 0]
                    x_list = [x_list[0], '']
                xaxis = np.arange(0, len(x_list), 1.0)
                w = float(xaxis[1] - xaxis[0]) / 4
                h = max(py) / 50

                fig = plt.figure(figsize=(16, 8))
                plt.bar(xaxis - w / 2, py, width=w, color='blue')
                plt.bar(xaxis + w / 2, fy, width=w, color='red')
                plt.xticks(xaxis, x_list)
                plt.title(keyword + ' result', fontsize=40)
                plt.ylim([0, 1.1 * max(max(fy), max(py)) + 5])
                plt.xlim([-1, max(xaxis) + 1])
                for i in range(len(py)):
                    plt.text(x=xaxis[i] - w, y=py[i] + h, s=py[i], fontsize=15, color='blue', )
                    plt.text(x=xaxis[i] + w / 2, y=fy[i] + h, s=fy[i], fontsize=15, color='red', )
                    plt.text(x=xaxis[i] - w, y=py[i] + 5 * h, s='{:0>2.2f}%'.format(100 * py[i] / (fy[i] + py[i])),
                             fontsize=15, color='green')

                if is_save:
                    plt.savefig(save_path + '/'+ keyword + '.png')
                else:
                    plt.show()
                continue

            if keyword == 'transpose':
                x_list = [12, 120, 201, 300]
            elif keyword == 'rotation':
                x_list = [0, 90, 180, 270, 360]

            else:
                maax = math.ceil(max(res_dict['pass'][keyword]))
                n = maax / n_step
                x_list = []
                for i in range(n_step + 1):
                    x_list.append(i * n)

            fys, fxs, fpatchs = plt.hist(res_dict['fail'][keyword], x_list, histtype='bar', rwidth=0.9, density=False,
                                         align='mid')
            pys, pxs, ppatchs = plt.hist(res_dict['pass'][keyword], x_list, histtype='bar', rwidth=0.9, density=False,
                                         align='mid')
            plt.close()

            x = []
            py, fy = [], []

            if keyword == 'transpose':
                fy = fys
                py = pys
                x = ['012', '120', '201']

            elif keyword == 'rotation':
                fy = fys
                py = pys
                x = ['0', '90', '180', '270']

            else:
                for i in range(len(fys)):
                    if (fys[i] + pys[i]) == 0:
                        continue
                    else:
                        py.append(pys[i])
                        fy.append(fys[i])
                        x.append(str(round(x_list[i], 1)) + '~{:0>1.1f}'.format(x_list[i] + maax / n_step))

            if len(py) < 2:
                py = [py[0], 0]
                fy = [fy[0], 0]
                x = [x[0], '']

            xaxis = np.arange(0, len(x), 1.0)
            w = float(xaxis[1] - xaxis[0]) / 4
            h = max(py) / 100

            plt.figure(figsize=(16, 8))
            plt.bar(xaxis - w / 2, py, width=w, color='blue')
            plt.bar(xaxis + w / 2, fy, width=w, color='red')
            plt.xticks(xaxis, x)
            plt.title(keyword + ' result', fontsize=40)

            for i in range(len(py)):
                plt.text(x=xaxis[i] - w, y=py[i] + h, s=int(py[i]), fontsize=15, color='blue')
                plt.text(x=xaxis[i], y=fy[i] + h, s=int(fy[i]), fontsize=15, color='red')
                plt.text(x=xaxis[i] - w, y=py[i] + 5 * h, s='{:0>2.2f}%'.format(100 * py[i] / (fy[i] + py[i])), fontsize=15,
                         color='green')

            plt.ylim([0, 1.1 * max(max(fy), max(py)) + 5])
            plt.xlim([-1, max(xaxis) + 1])

            if is_save:
                plt.savefig(save_path + '/' + keyword + '.png')
            else:
                plt.show()


    def plot_position(self,res_dict):
        '''
        plot the position variation of type result
        :param res_dict: dictionary from load_result
        '''

        # do kernel density estimation to get smooth estimate of distribution
        # make grid of points
        pf_list = ['pass','fail']
        for pf in pf_list:
            x = res_dict[pf]['x']
            y = res_dict[pf]['y']
            z = res_dict[pf]['z']

            # Define the borders
            deltaX = (max(x) - min(x)) / 10
            deltaY = (max(y) - min(y)) / 10
            deltaZ = (max(z) - min(z)) / 10
            xmin = min(x) - 2 * deltaX
            xmax = max(x) + deltaX
            ymin = min(y) - 2 * deltaY
            ymax = max(y) + deltaY
            zmin = min(y) - 4 * deltaZ
            zmax = max(y) + deltaZ
            print(xmin, xmax, ymin, ymax, zmin, zmax)
            # Create meshgrid
            xx, yy, zz = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]

            values = np.vstack([x, y, z])
            kernel = st.gaussian_kde(values)
            positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
            density = np.reshape(kernel(positions).T, xx.shape)

            # plot points
            ax = plt.subplot(projection='3d')
            ax.plot(x, y, z, 'o')

            # plot projection of density onto z-axis
            plotdat = np.sum(density, axis=2)
            plotdat = plotdat / np.max(plotdat)
            plotx, ploty = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            ax.contour(plotx, ploty, plotdat, offset=zmin, zdir='z')

            # This is new
            # plot projection of density onto y-axis
            plotdat = np.sum(density, axis=1)  # summing up density along y-axis
            plotdat = plotdat / np.max(plotdat)
            plotx, plotz = np.mgrid[xmin:xmax:100j, zmin:zmax:100j]
            ax.contour(plotx, plotdat, plotz, offset=ymax, zdir='y')

            # plot projection of density onto x-axis
            plotdat = np.sum(density, axis=0)  # summing up density along z-axis
            plotdat = plotdat / np.max(plotdat)
            ploty, plotz = np.mgrid[ymin:ymax:100j, zmin:zmax:100j]
            ax.contour(plotdat, ploty, plotz, offset=xmin, zdir='x')

            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_zlim((zmin, zmax))

            sns.jointplot(x=res_dict[pf]['x'], y=res_dict[pf]['y'], kind='kde').set_axis_labels("x", "y")
            plt.title('axial')
            sns.jointplot(x=res_dict[pf]['x'], y=res_dict[pf]['z'], kind='kde').set_axis_labels("x", "z")
            plt.title('coronal')
            plt.show()

class figure_type():
    def __init__(self):
        self.model_no = '37'
        self.gpu_num = list(range(FLAGS.num_gpu))
    
    def tf_model(self):
        # with tf.variable_scope(tf.get_variable_scope()):

        model = convmodel()
        self.image, self.is_training, self.prob, self.conv, self.feature = model.convnet_test()
        saver = tf.train.Saver(max_to_keep=1000)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver.restore(self.sess, FLAGS.save_path + '/model-37')

        # self. towers = {}
        # with tf.variable_scope(tf.get_variable_scope()):
        #     for gpu in self.gpu_num:
        #         with tf.device('/gpu:%d' % gpu):
        #             with tf.name_scope('%s_%d' % ('tower', gpu)) as scope:
        #                 num = str(gpu)
        #                 self.towers['img' + num], self.towers['label' + num], self.towers['loss' + num], \
        #                 self.towers['tacc' + num], self.towers['is_training' + num], self.towers['prob' + num], total_loss \
        #                     = tower_loss(scope)
        #                 tf.get_variable_scope().reuse_variables()
        #
        # saver = tf.train.Saver(max_to_keep=1000)
        # sess_config = tf.ConfigProto(allow_soft_placement=True)
        # sess_config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=sess_config)
        # saver.restore(self.sess, FLAGS.save_path + '/model-' + self.model_no)

    def print_weights(self,csv_path):
        '''
        print weights of the last layer
        ex) print weights('/data/dk')
        :param csv_path:dir for csv file
        '''
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            with tf.variable_scope("model1", reuse=True):
                w = tf.get_variable("fc1")
                arr = w.eval(session=self.sess)
                arr = np.squeeze(arr)
            f = open(csv_path+'/weight.csv', 'w')
            csvwriter = csv.writer(f)
            for row in arr:
                csvwriter.writerow(row)
            f.close()

        print arr

    def CAM(self,is_save=False):
        # show or save figs and class activation map of 3 types

        save_path = '/data/dk/images/type_final/valid/'
        path = '/data/dk/crops/train/TP/LIDC/LIDC-IDRI-0435/20000101/1.3.6.1.4.1.14519.5.2.1.6279.6001.146389062541391302265553091834/1.3.6.1.4.1.14519.5.2.1.6279.6001.404303213662391867606219459686/54_421_201_3.0_0.53125_1.0_0.53125_0_012.npy'
        img = np.load(path)
        input = np.expand_dims(img[8:40,...], axis =0)
        input = np.expand_dims(input, axis =4)

        if is_save:
            if not os.path.exists(save_path):
                os.makedirs(save_path+'/solid')
                os.makedirs(save_path+'/part-solid')
                os.makedirs(save_path+'/non-solid')

        # FC weights
        with tf.variable_scope("model1", reuse=True):
            w = tf.get_variable("fc1")
            arr = w.eval(session=self.sess)
            arr = np.squeeze(arr)
        w_nsn = arr[:,0]
        w_psn = arr[:,1]
        w_sn = arr[:,2]

        # top N weights
        weights_nsn = np.array(w_nsn)
        weights_psn = np.array(w_psn)
        weights_sn = np.array(w_sn)
        ind_nsn= weights_nsn.argsort()[-5:][::-1]
        ind_psn= weights_psn.argsort()[-5:][::-1]
        ind_sn= weights_sn.argsort()[-5:][::-1]

        path_dict = {'nsn':[], 'psn':[], 'sn':[]}

        # Last conv output
        cur_res = self.sess.run([self.prob,self.conv], feed_dict={self.image: input, self.is_training : False})
        conv_arr = cur_res[1]
        prob_arr = cur_res[0]

        # plot activation layer
        prob_out = prob_arr

        print 'nsn : %f / psn : %f /  sn : %f'%(prob_out[0],prob_out[1],prob_out[2])
        if prob_out.max() == prob_out[0]:
            img_path = save_path+'/non-solid'
            path_dict['nsn'].append(path)
        if prob_out.max() == prob_out[1]:
            img_path = save_path+'/part-solid'
            path_dict['psn'].append(path)
        if prob_out.max() == prob_out[2]:
            img_path = save_path+'/solid'
            path_dict['sn'].append(path)

        cur_path = path.split('/')[6]+'__'+path.split('/')[7]+'__'+path.split('/')[-1][:-4]
        cam_nsn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')
        cam_psn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')
        cam_sn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')

        print conv_arr.shape
        out_arr = conv_arr[0]

        # sum top N
        for j in ind_nsn:
            mat = w_nsn[j]*out_arr[:,:,:,j]
            cam_nsn += mat
        for j in ind_psn:
            mat = w_psn[j]*out_arr[:,:,:,j]
            cam_psn += mat
        for j in ind_sn:
            mat = w_sn[j]*out_arr[:,:,:,j]
            cam_sn += mat

        cam_nsn = self.CAM_reshape(cam_nsn)
        cam_psn = self.CAM_reshape(cam_psn)
        cam_sn = self.CAM_reshape(cam_sn)

        fig, ax = plt.subplots(4,4, figsize =(10,10))
        for num_img in range(4):
            ax[num_img / 2, 2+num_img % 2].imshow(cam_nsn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
            ax[num_img / 2, 2+num_img % 2].axis('off')
        in_img = img
        for num_img in range(4):
            ax[num_img / 2, num_img % 2].imshow(in_img[23+num_img, ...], cmap='bone', clim=(0., 1.5))
            ax[num_img / 2, num_img % 2].axis('off')
        for num_img in range(4):
            ax[2+num_img / 2, num_img % 2].imshow(cam_psn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
            ax[2+num_img / 2, num_img % 2].axis('off')
        for num_img in range(4):
            ax[2+num_img / 2, 2+num_img % 2].imshow(cam_sn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
            ax[2+num_img / 2, 2+num_img % 2].axis('off')

        ax[0, 0].text(1, 4, 'non-solid : %1.3f' % prob_out[0], color='white')
        ax[0, 0].text(1, 8, 'part-solid : %1.3f' % prob_out[1], color='white')
        ax[0, 0].text(1, 12, 'solid : %1.3f' % prob_out[2], color='white')

        ax[0, 2].text(1, 4, '<non-solid>', color='white')
        ax[2, 0].text(1, 4, '<part-solid>', color='white')
        ax[2, 2].text(1, 4, '<solid>', color='white')

        if is_save:
            plt.savefig(img_path + '/' + cur_path + '.png')
        else:
            plt.show()
        plt.close()


    def CAM_all(self,is_save=False):
        # show or save figs and class activation map of 3 types
        # is_type, is_save = 'solid', True
        type_list = ['non-solid','part-solid','solid']

        for is_type in type_list:
            print '<%s>'%is_type

            val_lists = ['/data/dk/type_crops/valid/'+is_type+'/*/*/*/*/*/','/data/dk/type_crops/valid2/'+is_type+'/*/*/*/*/*/']
            save_path = '/data/dk/images/type_final/valid_'+is_type+'_all/'

            # path = data_load(val_lists)
            path = original_load(val_lists)
            input = npy_load(path)


            batch = 30

            if is_save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path+'/solid')
                    os.makedirs(save_path+'/part-solid')
                    os.makedirs(save_path+'/non-solid')

            total_data = len(input)
            print 'input :', total_data

            # FC weights
            with tf.variable_scope("model1", reuse=True):
                w = tf.get_variable("fc1")
                arr = w.eval(session=self.sess)
                arr = np.squeeze(arr)
            w_nsn = arr[:,0]
            w_psn = arr[:,1]
            w_sn = arr[:,2]

            # top N weights
            weights_nsn = np.array(w_nsn)
            weights_psn = np.array(w_psn)
            weights_sn = np.array(w_sn)
            ind_nsn= weights_nsn.argsort()[-5:][::-1]
            ind_psn= weights_psn.argsort()[-5:][::-1]
            ind_sn= weights_sn.argsort()[-5:][::-1]

            path_dict = {'nsn':[], 'psn':[], 'sn':[]}

            # Last conv output
            for k in range(0,total_data,batch):
                b_start = k
                if b_start+batch >= total_data:
                    b_end = total_data
                else:
                    b_end = k+batch
                print 'batch : %d ~ %d'%(b_start, b_end-1)
                cur_res = self.sess.run([self.prob,self.conv], feed_dict={self.image: input[b_start:b_end], self.is_training : False})
                conv_arr = cur_res[1]
                prob_arr = cur_res[0]

                # plot activation layer
                for i in range(len(prob_arr)):
                    if b_start == b_end - 1:
                        prob_out = prob_arr
                    else:
                        prob_out = prob_arr[i]
                    print 'nsn : %f / psn : %f /  sn : %f'%(prob_out[0],prob_out[1],prob_out[2])
                    if prob_out.max() == prob_out[0]:
                        img_path = save_path+'/non-solid'
                        path_dict['nsn'].append(path[b_start+i])
                    if prob_out.max() == prob_out[1]:
                        img_path = save_path+'/part-solid'
                        path_dict['psn'].append(path[b_start+i])
                    if prob_out.max() == prob_out[2]:
                        img_path = save_path+'/solid'
                        path_dict['sn'].append(path[b_start+i])

                    cur_path = path[i+b_start].split('/')[6]+'__'+path[i+b_start].split('/')[7]+'__'+path[i+b_start].split('/')[-1][:-4]
                    cam_nsn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')
                    cam_psn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')
                    cam_sn = np.zeros([conv_arr.shape[1], conv_arr.shape[2], conv_arr.shape[3]], dtype='float32')

                    out_arr = conv_arr[i]

                    # sum top N
                    for j in ind_nsn:
                        mat = w_nsn[j]*out_arr[:,:,:,j]
                        cam_nsn += mat
                    for j in ind_psn:
                        mat = w_psn[j]*out_arr[:,:,:,j]
                        cam_psn += mat
                    for j in ind_sn:
                        mat = w_sn[j]*out_arr[:,:,:,j]
                        cam_sn += mat

                    cam_nsn = self.CAM_reshape(cam_nsn)
                    cam_psn = self.CAM_reshape(cam_psn)
                    cam_sn = self.CAM_reshape(cam_sn)

                    fig, ax = plt.subplots(4,4, figsize =(10,10))
                    for num_img in range(4):
                        ax[num_img / 2, 2+num_img % 2].imshow(cam_nsn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
                        ax[num_img / 2, 2+num_img % 2].axis('off')
                    in_img = input[i+b_start,:,:,:,0]
                    for num_img in range(4):
                        ax[num_img / 2, num_img % 2].imshow(in_img[15+num_img, ...], cmap='bone', clim=(0., 1.5))
                        ax[num_img / 2, num_img % 2].axis('off')
                    for num_img in range(4):
                        ax[2+num_img / 2, num_img % 2].imshow(cam_psn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
                        ax[2+num_img / 2, num_img % 2].axis('off')
                    for num_img in range(4):
                        ax[2+num_img / 2, 2+num_img % 2].imshow(cam_sn[15+num_img, ...])#, cmap='hot', clim=(0., 1.5))
                        ax[2+num_img / 2, 2+num_img % 2].axis('off')

                    ax[0, 0].text(1, 4, 'non-solid : %1.3f' % prob_out[0], color='white')
                    ax[0, 0].text(1, 8, 'part-solid : %1.3f' % prob_out[1], color='white')
                    ax[0, 0].text(1, 12, 'solid : %1.3f' % prob_out[2], color='white')

                    ax[0, 2].text(1, 4, '<non-solid>', color='white')
                    ax[2, 0].text(1, 4, '<part-solid>', color='white')
                    ax[2, 2].text(1, 4, '<solid>', color='white')


                    if is_save:
                        plt.savefig(img_path + '/' + cur_path + '.png')
                    else:
                        plt.show()
                    plt.close()

                    if b_start == b_end - 1:
                        break

            if is_save:
                with open(save_path+ '/path_dict.json', 'w') as outfile:
                    json.dump(path_dict, outfile)


    def CAM_reshape(self,cam_img):
        cam_img = zoom(cam_img,(8,8,8))
        cam_img = cam_img - np.min(cam_img)
        cam_img = cam_img/np.max(cam_img)
        cam_img = np.uint8(255*cam_img)
        cam_img = gaussian_filter(cam_img,10)

        return cam_img


    def feature_map(self):
        '''
        json file : json from CAM_all
        :return:
        '''
        is_type = 'non-solid'
        is_save = False
        json_path = '/data/dk/images/type_final/valid_'+is_type+'_all'
        save_path = '/data/dk/images/feature_SEdense_37/valid_'+is_type

        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path+'/nsn')
            os.mkdir(save_path+'/psn')
            os.mkdir(save_path+'/sn')

        with open(json_path+ '/path_dict.json') as json_file:
            res_dict = json.load(json_file)

        # model = convmodel()
        # image, label, loss, acc, is_training, prob, conv, feature = model.convnet_test()
        # saver = tf.train.Saver(max_to_keep=1000)
        #
        # sess_config = tf.ConfigProto(allow_soft_placement=True)
        # sess_config.gpu_options.allow_growth = True
        # sess = tf.Session(config=sess_config)
        #
        # saver.restore(sess, FLAGS.save_path + '/weights/model-37')

        type_list = ['nsn','psn','sn']

        for type in type_list:
            print 'number of %s : %d'%(type,len(res_dict[type]))

        for type in type_list:
            path_list = res_dict[type]
            input = npy_load(path_list)
            batch = 30

            total_data = len(input)
            print type,'input :', total_data
            # if type == 'sn':
            #     v_label = 2*np.ones([total_data, 1])
            #     col = 2
            # elif type == 'psn':
            #     v_label = np.ones([total_data,1])
            #     col = 1
            # elif type == 'nsn':
            #     v_label = np.zeros([total_data, 1])
            #     col = 0
            # else: print 'wrong type'

            # Last feature
            for k in range(0,total_data,batch):
                b_start = k
                if k+batch >= total_data:
                    b_end = total_data
                else:
                    b_end = k+batch
                print 'batch : %d ~ %d'%(b_start, b_end-1)
                cur_res = self.sess.run([self.prob,self.feature], feed_dict={self.image: input[b_start:b_end], self.is_training : False})
                feat_arr = cur_res[1]
                prob_out = cur_res[0]

                for j in range(len(feat_arr)):
                    cur_feat = feat_arr[j]
                    cur_feat = np.squeeze(cur_feat)
                    cur_prob = prob_out[j]
                    print 'nsn : %f / psn : %f /  sn : %f'%(cur_prob[0],cur_prob[1],cur_prob[2])

                    cur_feat = gaussian_filter(cur_feat,10)
                    plt.plot(cur_feat)
                    if is_save:
                        img_path = save_path+'/'+type
                        cur_path = path_list[j + b_start].split('/')[6] + '__' + path_list[j + b_start].split('/')[7] + '__' + \
                                   path_list[j + b_start].split('/')[-1][:-4]
                        plt.savefig(img_path+'/'+cur_path+'.png')
                    else: plt.show()
                    plt.close()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_VISIBLE_DEVICES

    if FLAGS.mode == 'histograim':
        at = analysis_type()
        res_dict = at.load_result('/data/dk/result/type_test/')
        at.plot_histogram(['rotation', 'transpose', 'interval', 'thickness', 'iskind', 'mA', 'KVP', 'mAs',
                                'convolutionKernel', 'diameter'],res_dict)
    elif FLAGS.mode == 'position':
        at = analysis_type()
        res_dict = at.load_result('/data/dk/result/type_test/')
        at.plot_position(res_dict)
    elif FLAGS.mode == 'CAM':
        ft = figure_type()
        ft.tf_model()
        ft.CAM()
    elif FLAGS.mode == 'CAM_all':
        ft = figure_type()
        ft.tf_model()
        ft.CAM_all()
    elif FLAGS.mode == 'feature_map':
        ft = figure_type()
        ft.tf_model()
        ft.feature_map()

if __name__ == '__main__':
    main()
from glob import glob
import numpy as np
from large_model_type import convmodel
import tensorflow as tf

def data_load(*args):

    dirs = []
    for idx, arg in enumerate(args):
        if isinstance(arg, list):
            for cur in arg:
                if cur[-1] == '/': cur = cur[:-1]
                # dirs += glob(cur + '/*_0_012.npy')
                dirs += glob(cur + '/*.npy')

        else:
            if arg[-1] == '/': arg = arg[:-1]
            # dirs += glob(arg + '/*_0_012.npy')
            dirs += glob(arg + '/*.npy')

    if len(dirs) == 0:
        print "Please check dirs (len(dir)==0)"
        raw_input("Enter")

    # dirs_new = []
    #
    # for i in dirs:
    #     if i.split('_')[-4] == '1.6' or i.split('_')[-4] == '1.8':
    #         continue
    #     else:
    #         dirs_new.append(i)
    #
    #
    # return dirs_new
    return dirs


def original_load(*args):

    dirs = []
    for idx, arg in enumerate(args):
        if isinstance(arg, list):
            for cur in arg:
                if cur[-1] == '/': cur = cur[:-1]
                dirs += glob(cur + '/*_0_012.npy')

        else:
            if arg[-1] == '/': arg = arg[:-1]
            dirs += glob(arg + '/*_0_012.npy')

    if len(dirs) == 0:
        print "Please check dirs (len(dir)==0)"
        raw_input("Enter")

    dirs_new = []

    for i in dirs:
        if i.split('_')[-4] == '1.0':
            dirs_new.append(i)

    return dirs_new


def npy_load(npy_path):

    # x = np.empty(shape=[len(npy_path), 24, 40, 40])
    x = np.empty(shape=[len(npy_path), 32, 48, 48])
    # x = np.empty(shape=[len(npy_path), 48, 48, 48])
    for idx, cur_path in enumerate(npy_path):
        arr = np.load(cur_path)
        x[idx] = arr[8:40,:,:]
        # x[idx] = arr[12:36,4:44,4:44]

    return np.expand_dims(x, axis=4)


def tower_loss(scope):

    model = convmodel()
    image, label, loss, acc, is_training, prob = model.convnet()

    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    return image, label, loss, acc, is_training, prob, total_loss

def tower_test(scope):

    model = convmodel()
    image, is_training, prob, out1, out2 = model.convnet_test()

    return image, is_training, prob

def average_gradients(tower_grads):
    """Calculate the average gradient
       for each shared variable across all towers

       Args:
           tower_grads : List of lists of (gradient, variable) tuples.
           The outer list is over individual gradients.
           The inner list is over the gradient calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # print grad_and_vars
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, axis=0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared across towers.
        # So .. we will just return the first tower's pointer to the Variable
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)

    return average_grads

def send_email(Subject, Text, From='kdkhome@naver.com', To='sunnmoon137@gmail.com'):

    import smtplib
    from email.mime.text import MIMEText

    pwd = '~!Q@W#E$R'
    msg = MIMEText(Text, _charset='euc-kr')
    msg['Subject'] = Subject
    msg['From'] = From
    msg['To'] = To

    try:
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(From, pwd)
        server.sendmail(From, To, msg.as_string())
        server.quit()
    except smtplib.SMTPException as e:
        print e

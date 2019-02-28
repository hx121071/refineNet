import tensorflow as tf
import numpy as np
import data
import datetime
from matplotlib import pyplot as plt
import cv2
import os
from data import Roidata
import config as cfg
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from timer import Timer
from bbox_transform import bbox_transform_inv,clip_boxes
slim = tf.contrib.slim


def decoder_arg_scope(weight_decay = 0.0001, stddev = 0.1, is_training = True, batch_norm_var_collection = 'moving_vars'):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training,
        'variables_collections':{
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }
    with slim.arg_scope([slim.conv2d], weights_regularizer = slim.l2_regularizer(weight_decay),
                        weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                        activation_fn = tf.nn.relu,
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = batch_norm_params) as sc:
        return sc

# def small_net(inputs, keep_prob, is_training = True, scope = None):
#     with tf.variable_scope(scope, 'refine_net', [inputs]):
#         with slim.arg_scope(decoder_arg_scope(is_training = is_training)):
#             # block1
#             net = slim.conv2d(inputs, 32, [3, 3], scope = "conv1_1")
#             net = slim.conv2d(net, 32, [3, 3], scope = 'conv1_2')
#             net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool1")
#             # block2
#             net = slim.conv2d(net, 64, [3, 3], scope = "conv2_1")
#             net = slim.conv2d(net, 64, [3, 3], scope = "conv2_2")
#             net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool2")
#             # block3
#             net = slim.conv2d(net, 128, [3, 3], scope = "conv3_1")            
#             net = slim.conv2d(net, 128, [3, 3], scope = "conv3_2")           
#             net = slim.conv2d(net, 128, [3, 3], scope = "conv3_3")
#             net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool3")
#             # block4          
#             net = slim.conv2d(net, 256, [3, 3], scope = "conv4_1")           
#             net = slim.conv2d(net, 256, [3, 3], scope = "conv4_2")            
#             net = slim.conv2d(net, 256, [3, 3], scope = "conv4_3")
#             net = slim.avg_pool2d(net, 7, scope='gap_5')
#             net = slim.flatten(net, scope='flat_6')
#             print(net.shape)
#             net = slim.fully_connected(net, 256, scope='fc_7')
#             print(net.shape)
#             net = slim.fully_connected(net, 4096, scope='fc_8')
#             net = slim.dropout(
#                 net, keep_prob=keep_prob, is_training=is_training,
#                 scope='dropout_9')
#             net = slim.fully_connected(
#                 net, cfg.num_outputs, activation_fn=None, scope='fc_10')
#             return net


def small_net(inputs, keep_prob, is_training = True, scope = None):
    with tf.variable_scope(scope, 'refine_net', [inputs]):
        with slim.arg_scope(decoder_arg_scope(is_training = is_training)):
            # block1
            net = slim.conv2d(inputs, 32, [3, 3], scope = "conv1_1")
            net = slim.max_pool2d(net, [3, 3], stride = 2, padding = "SAME", scope = "pool1")
            # block2
            net = slim.conv2d(net, 64, [3, 3], scope = "conv2_1")
            net = slim.max_pool2d(net, [3, 3], stride = 2, padding = "SAME", scope = "pool2")
            # block3
            net = slim.conv2d(net, 128, [3, 3], scope = "conv3_1")            
            net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool3")
            # block4          
            net = slim.conv2d(net, 256, [3, 3], scope = "conv4_1")           
            net = slim.avg_pool2d(net, 7, scope='gap_5')
            net = slim.flatten(net, scope='flat_6')
            print(net.shape)
            net = slim.fully_connected(net, 512, scope='fc_7')
            print(net.shape)
            # net = slim.fully_connected(net, 4096, scope='fc_8')
            net = slim.dropout(
                net, keep_prob=keep_prob, is_training=is_training,
                scope='dropout_8')
            net = slim.fully_connected(
                net, cfg.num_outputs, activation_fn=None, scope='fc_9')
            return net

def smooth_l1_loss(bbox_pred, bbox_targets, sigma):
    """
    ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
    SmoothL1(x) = 0.5 * (sigma * x)^2, if |x| < 1 / sigma^2
                    |x| - 0.5 / sigma^2, otherwise
    """
    sigma2 = sigma * sigma
    diff = tf.abs(bbox_targets - bbox_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0 / sigma2), tf.float32)
    loss = (less_than_one * 0.5 * diff**2 * sigma2) + (1 - less_than_one)*(diff - 0.5 / sigma2)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    # tf.add_to_collection("losses", loss)
    tf.losses.add_loss(loss)
    tf.summary.scalar('smooth_l1_loss', loss)
    return loss

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    roidata = Roidata()

    # inputs data 
    img_ph = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.image_size, cfg.image_size, 1])
    labels_ph = tf.placeholder(tf.float32, shape=[cfg.batch_size, 4])
    logits = small_net(img_ph, 0.5)
   

    #loss 
    loss = smooth_l1_loss(logits, labels_ph, 2)    
    # print(type(tf.get_collection("losses")))
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('total_loss', total_loss)
    
    # summary operation
    output_dir = ('samller_output_calibration')
    ckpt_file  = os.path.join(output_dir, 'refine')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(output_dir, flush_secs=60)

    # optimize operation
    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(
        cfg.learning_rate, global_step, cfg.decay_steps,
        cfg.decay_rate, True, name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, global_step=global_step)
    train_op = slim.learning.create_train_op(
        total_loss, optimizer, global_step=global_step
    )
    
    # session and saver
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 1.0       
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    
    writer.add_graph(sess.graph)
    
    train_timer = Timer()
    for step in range(1, cfg.max_iter+1):
        load_timer = Timer()
        load_timer.tic()
        img, labels = roidata.get()
        load_timer.toc()

        feed_dict = {img_ph:img, labels_ph:labels}
        if step % cfg.summary_iter == 0:
                if step % (cfg.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, total_loss_val, loss_val, _ = sess.run(
                        [summary_op, total_loss, loss, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {:.5f}, Total_loss: {:5.3f}, smooth_l1_loss: {:5.3f}\nSpeed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        roidata.epoch,
                        int(step),
                        round(learning_rate.eval(session=sess), 6),
                        total_loss_val,
                        loss_val,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, cfg.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = sess.run(
                        [summary_op, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                writer.add_summary(summary_str, step)

        else:
            train_timer.tic()
            sess.run(train_op, feed_dict=feed_dict)
            train_timer.toc()

        if step % cfg.save_iter == 0:
            print('{} Saving checkpoint file to: {}'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                output_dir))
            saver.save(
                sess, ckpt_file, global_step=global_step)
        if step == cfg.max_iter:
            print('{} Saving checkpoint file to: {}'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                output_dir))
            saver.save(
                sess, ckpt_file, global_step=global_step)


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    roidata = Roidata(is_train=False, with_keypoints=False)
    img_ph = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.image_size, cfg.image_size, 1])


    logits = small_net(img_ph, 1.0, is_training=False)

    ckpt_file = 'samller_output_calibration/refine-31970'
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 1.0     
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_file)

    result_file = open('zebrish_yolo_143000_refine_smaller_with_gap_calibration.txt', 'w')
    test_timer = Timer()
    img, im_path, proposal, gt_boxes, score = roidata.get()
        # print(proposal, gt_boxes)
    feed_dict = {img_ph:img}
    # for i in range(100):
    #     logits_val = sess.run(logits, feed_dict=feed_dict)
    for i in range(len(roidata.data)):
        
        # if score < 0.5:
        img, im_path, proposal, gt_boxes, score = roidata.get()
        # print(proposal, gt_boxes)
        feed_dict = {img_ph:img}
        # cv2.imshow("origin", img[0])
        test_timer.tic()
        logits_val = sess.run(logits, feed_dict=feed_dict)
        test_timer.toc()
        logits_val = logits_val * np.array(cfg.BBOX_NORMALIZE_STDS)
        proposalnp = np.array([proposal], dtype=np.float32)
        pred_gt = bbox_transform_inv(proposalnp, logits_val)
        
        print('Average detecting time: {:.4f}s'.format(
            test_timer.average_time))
        origin_img = cv2.imread(im_path)
        # # print(im_path)
        im_index = im_path.split('/')[-1][:-4]
        # # print(im_index)
        print(pred_gt)
        pred_gt = clip_boxes(pred_gt, [origin_img.shape[0], origin_img.shape[1]])[0].astype(np.int32)
        
        result_file.write('{:s} {:.4f} {:d} {:d} {:d} {:d}\n'.format(
            im_index, score, pred_gt[0], pred_gt[1], pred_gt[2], pred_gt[3]))


        cv2.rectangle(origin_img, (proposal[0], proposal[1]), (proposal[2], proposal[3]), (255, 0, 0))
        cv2.rectangle(origin_img, (gt_boxes[0], gt_boxes[1]), (gt_boxes[2], gt_boxes[3]), (0, 255, 0))
        cv2.rectangle(origin_img, (pred_gt[0], pred_gt[1]), (pred_gt[2], pred_gt[3]), (0, 0, 255))
        cv2.imshow("test", origin_img)
        cv2.waitKey(0)

    


if __name__ == '__main__':
    # train()
    test()
    # labels = np.array([[-0.05063291, -0.4620462,   1.2283633,   1.1190169 ]])
    # logits = np.array([[-0.11074541,  0.02087769, -0.07148118, 0.15149847]])
    # diff = np.abs(labels - logits)
    # print(diff)
import tensorflow as tf 
def smooth_l1_loss(bbox_pred, bbox_targets):
    """
    ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
    SmoothL1(x) = 0.5 * (sigma * x)^2, if |x| < 1 / sigma^2
                    |x| - 0.5 / sigma^2, otherwise
    """
    diff = tf.abs(bbox_targets - bbox_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one)*(diff - 0.5)
    return loss


# if __name__ == '__main__':

import os


batch_size = 1

BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

image_size = 56

num_outputs = 4

learning_rate = 0.001
decay_rate = 0.1
decay_steps = 10007 * 6

max_iter = 10007 * 10

summary_iter = 10

save_iter = 10007

GPU = '0'
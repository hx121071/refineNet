import os


batch_size = 32

BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

image_size = 56

num_outputs = 4

learning_rate = 0.001
decay_rate = 0.1
decay_steps = 3197 * 6

max_iter = 3197 * 10

summary_iter = 10

save_iter = 3197

GPU = '0'
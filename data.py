import os 
import xml.etree.ElementTree as ET 
import numpy as np 
import pickle 
import config as cfg
from bbox_transform import bbox_transform,bbox_transform_inv
import cv2

# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     size = tree.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)

#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [int(bbox.find('xmin').text),
#                               int(bbox.find('ymin').text),
#                               int(bbox.find('xmax').text),
#                               int(bbox.find('ymax').text)]
#         obj_struct['width'] = w 
#         obj_struct['height'] = h
#         objects.append(obj_struct)

#     return objects
def parse_rec_with_keypoint(filename, im_path):
    objects = []
    # print(im_path)
    gt_keypoints = np.loadtxt(filename).astype(np.int32)
    obj_struct = {}
    # boxes
    if gt_keypoints.shape[0] == 22:
        gt_keypoints = gt_keypoints[1:, :] # 21 * 2
        x1 = np.min(gt_keypoints[:, 0])
        y1 = np.min(gt_keypoints[:, 1])
        x2 = np.max(gt_keypoints[:, 0])
        y2 = np.max(gt_keypoints[:, 1])

        im = cv2.imread(im_path)
        h,w,_ = im.shape
        dw1 = int((x2 - x1) * 0.15)
        dh1 = int((y2 - y1) * 0.15)
        x1 = max(x1 - dw1, 0)
        y1 = max(y1 - dh1, 0)
        x2 = min(x2 + dw1, w-1)
        y2 = min(y2 + dh1, h-1)
        
        boxes = [ x1, y1, x2, y2]
        obj_struct['difficult'] = 0
    else:
        obj_struct['difficult'] = 1
        boxes = [0, 0, 0, 0]
    obj_struct['bbox'] = boxes
    objects.append(obj_struct)
    # print(objects)
    return objects
def cal_IOU(gt, pre):
    x1, y1, x2, y2 = pre
    ixmin = np.maximum(gt[0], int(x1))
    iymin = np.maximum(gt[1], int(y1))
    ixmax = np.minimum(gt[2], int(x2))
    iymax = np.minimum(gt[3], int(y2))
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((int(x2) - int(x1) + 1.) * (int(y2) - int(y1) + 1.) +
            (gt[2] - gt[0] + 1.) *
            (gt[3] - gt[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

class Roidata(object):


    def __init__(self, is_train=True, rebuild=False, with_keypoints = True):
        self.devkit_path = 'VOCdevkit'
        self.data_path = os.path.join(self.devkit_path, 'VOC2007')
        self.anno_pre = os.path.join(self.data_path, 'Keypoints', '{:s}.txt')
        self.img_pre = os.path.join(self.data_path, 'JPEGImages', '{:s}.jpg')
        self.cache_file = 'zebrish_yolo_143000_refine_smaller_with_gap.pkl'
        self.proposal_file = 'zebrish_yolo_143000_refine_smaller_with_gap.txt'
        self.batch_size = cfg.batch_size
        self.cursor = 0
        self.rebuild = rebuild
        self.with_keypoints = with_keypoints
        self.epoch = 1
        self.is_train = is_train
        self.data = []
        self.prepare()
        self.data_size = len(self.data)

    def get(self):
        
        # inputs to network
        img = np.zeros((cfg.batch_size, cfg.image_size, cfg.image_size, 1), dtype=np.float32)
        labels = np.zeros((cfg.batch_size, 4), dtype=np.float32)

        count = 0

        while count < cfg.batch_size:
            # print(count)
            im_path = self.data[self.cursor]['img_path']
            # print("path", im_path)
            targets = self.data[self.cursor]['targets']
            
            proposal = self.data[self.cursor]['proposal']
            gt_boxes = self.data[self.cursor]['gt_boxes']
            score = self.data[self.cursor]['score']
            img[count, :, :, :] = self.read_img(im_path, proposal)
            labels[count, :] =  targets
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.data):
                np.random.shuffle(self.data)
                self.cursor = 0
                self.epoch += 1
        if self.is_train:
            return img, labels
        else:
            return img, im_path, proposal, gt_boxes, score

    def read_img(self, img_path, proposal):
        # print(img_path)
        # print(proposal)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = proposal
        # print(x1, x2, y1, y2) 
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,225))
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = img[y1:y2, x1:x2]

        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = cv2.resize(img, (cfg.image_size, cfg.image_size)).astype(np.float32)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = img - 103.9
        img = np.reshape(img, (img.shape[0], img.shape[1], -1))
        return img
    def prepare(self):

        if os.path.isfile(self.cache_file) and not self.rebuild:
            print('Loading data from: ', self.cache_file)
            with open(self.cache_file, 'rb') as f:
                 self.data = pickle.load(f)
                 return
        
        print('Processing data from: ', self.data_path)

        f = open(self.proposal_file,  'r')

        proposal_records = [line.strip().split(' ') for line in f.readlines()]

        
        for proposal_record in proposal_records:
            im_index, score, x1, y1, x2, y2 = proposal_record
            # if float(score) < 1.0:
            single_data = {}
            img_path = self.img_pre.format(im_index)
            if self.with_keypoints:
                anno = self.anno_pre.format(im_index)
                anno_parse = parse_rec_with_keypoint(anno, img_path)[0]
                gt_boxes = anno_parse['bbox']
            else:
                gt_boxes = [0,0,1,1]
            # import pdb 
            # pdb.set_trace()
            # w = anno_parse['width']
            # h = anno_parse['height']
            # gt_boxes[0] = max(gt_boxes[0] - 20, 0)
            # gt_boxes[1] = max(gt_boxes[1] - 20, 0)
            # gt_boxes[2] = min(gt_boxes[2] + 20, w-1)
            # gt_boxes[3] = min(gt_boxes[3] + 20, h-1)
            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)
            x2 = max(int(x2), 0)
            y2 = max(int(y2), 0)
            proposal = [int(x1), int(y1), int(x2), int(y2)]
            IOU = cal_IOU(gt_boxes, proposal)
            if IOU < 0.9:
                targets = bbox_transform(np.array([proposal], dtype=np.float32), \
                                            np.array([gt_boxes], dtype=np.float32))
                targets = targets / np.array(cfg.BBOX_NORMALIZE_STDS)
                single_data['img_path'] = img_path
                single_data['gt_boxes'] = gt_boxes
                single_data['proposal'] = proposal 
                single_data['targets'] = targets 
                single_data['score'] = float(score)
                self.data.append(single_data)
        
        np.random.shuffle(self.data)

        print('Saving data to: ', self.cache_file)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print('Done prepare data')
        return

if __name__ == '__main__':
    roidata = Roidata()
    data = roidata.data[0]
    print(data)
    proposalnp = np.array([data['proposal']], dtype=np.float32)
    gt_boxesnp = np.array([data['gt_boxes']], dtype=np.float32)
    # print(proposalnp, gt_boxesnp)
    targets = bbox_transform(proposalnp, gt_boxesnp)
    print(targets)
    # targets = targets / np.array(cfg.BBOX_NORMALIZE_STDS)
    targets = data['targets'] * np.array(cfg.BBOX_NORMALIZE_STDS)
    pred_gt = bbox_transform_inv(proposalnp, targets)
    print(pred_gt)
    # print(targets)
    print(len(roidata.data))
    # for i in range(roidata.data_size):
    #     img, labels = roidata.get()
    # print(img.shape, labels.shape)
        

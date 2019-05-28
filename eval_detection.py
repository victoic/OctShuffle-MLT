from nms import get_boxes
import torch, os
import argparse

import numpy as np
import cv2

import data_gen
from data_gen import draw_box_points
import net_utils
from models import OctShuffleMLT

import unicodedata as ud

from Levenshtein import distance
import pandas as pd

from ocr_utils import print_seq_ext
import Polygon

def load_annotation(filelist):
  path_to_dir = filelist.split(os.path.basename(filelist))[0]
  list_file = open(filelist, 'r')
  gt_images = []
  gt_boxes = []
  fns = []

  for line in list_file:
    if len(gt_images) % 100 == 0:
      print(len(gt_images))
    file = os.path.split(line)
    path = path_to_dir+file[0]
    fns.append(file[0])
    gt_file = os.path.splitext('gt_'+file[1].split('\n')[0])[0]+".txt"
    gt_path = os.path.join(path,gt_file)
    with open(gt_path, 'r') as gt:
      boxes = []
      found = False
      for line_gt in gt:
        split_line = line_gt.replace('\ufeff','').split(',')
        if split_line[-1] != "###":
          found = True;
          split_line = [float(cell) for cell in split_line[0:8]]
          points = np.array(split_line).reshape(4,-1)
          boxes.append(points)
      if found:
        gt_image_path = os.path.join(path_to_dir,line.rstrip())
        gt_image = cv2.imread(gt_image_path)
        gt_images.append(gt_image)
        gt_boxes.append(boxes)
  return np.array(gt_images), np.array(gt_boxes), fns

def eval_detection(opts, net=None):
  if net == None:
    net = OctShuffleMLT(attention=True)
    net_utils.load_net(opts.model, net)
    if opts.cuda:
      net.cuda()

  images, gt_boxes, fns = load_annotation(opts.eval_list)  
  true_positives = 0
  false_positives = 0
  false_negatives = 0

  result_path = "/test_results"
  
  for i in range(images.shape[0]):
    image = np.expand_dims(images[i], axis=0)
    cp_image = image
    image_boxes_gt = np.array(gt_boxes[i])

    im_data = net_utils.np_to_variable(image, is_cuda=opts.cuda).permute(0, 3, 1, 2)
    seg_pred, rboxs, angle_pred, features = net(im_data)
    
    rbox = rboxs[0].data.cpu()[0].numpy()
    rbox = rbox.swapaxes(0, 1)
    rbox = rbox.swapaxes(1, 2)
    angle_pred = angle_pred[0].data.cpu()[0].numpy()
    segm = seg_pred[0].data.cpu()[0].numpy()
    segm = segm.squeeze(0)

    boxes =  get_boxes(segm, rbox, angle_pred, opts.segm_thresh)

    if (opts.debug):
      print(boxes.shape)
      print(image_boxes_gt.shape)
      print(image_boxes_gt.shape[0] == boxes.shape[0])
      print("============")

    false_positives += boxes.shape[0]
    false_negatives += image_boxes_gt.shape[0]
    for box in boxes:
      b = box[0:8].reshape(4,-1)
      poly = Polygon.Polygon(b)
      color=(239,19,19)
      for box_gt in image_boxes_gt:
        b_gt = box_gt[0:8].reshape(4,-1)
        poly_gt = Polygon.Polygon(b_gt)
        intersection = poly_gt | poly
        union = poly_gt & poly
        iou = (intersection.area()+1.0) / (union.area()+1.0)-1.0
        if iou > 0.5:
          color = (255,120,255)
          true_positives+=1
          false_negatives-=1
          false_positives-=1
          image_boxes_gt = np.array([bgt for bgt in image_boxes_gt if not np.array_equal(bgt, box_gt)])
          break
      cp_image = cv2.polylines(cp_image, [np.array(b, np.int32)], True, color)
    if (image_boxes_gt.shape[0] == boxes.shape[0]):
      dir_path = os.path.join(result_path, "successes/")
    else:
      dir_path = result_path
    file_name = fns[i]+".png"
    cv2.imwrite(os.path.join(dir_path, file_name), cp_image)

  print("tp: {} fp: {} fn: {}".format(true_positives, false_positives, false_negatives))
  precision = true_positives / (true_positives+false_positives)
  recall = true_positives / (true_positives+false_negatives)
  f_score = 2*precision*recall/(precision+recall)
  print("PRECISION: {} \t RECALL: {} \t F SCORE: {}".format(precision, recall, f_score))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-eval_list', default='dataset/TEST/test.txt')
  parser.add_argument('-segm_thresh', default=0.5)
  parser.add_argument('-model', default='backup/MobileMLT_200000.h5')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-input_size', type=int, default=256)
  parser.add_argument('-geo_type', type=int, default=0)
  parser.add_argument('-base_lr', type=float, default=0.0001)

  args = parser.parse_args()
  eval_detection(args)
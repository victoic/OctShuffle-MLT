'''
Created on Aug 31, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import torch
from torch.autograd import Variable

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v

def load_net(fname, net, optimizer=None, load_shared=True, load_ocr=True, load_detection=True, load_optimizer=True, reset_step=False):

  shared_layers = ['layer0', 'layer0_1']
  ocr_layers = ['conv5', 'conv6', 'conv7', 'conv8', 'conv9_1', 'conv9_2', 'conv10_s', 'conv11', 'batch5', 
                        'batch6', 'batch7', 'batch8', 'batch9', 'batch10_s']
  detection_layers = ['layer1', 'layer2', 'layer3', 'layer4', 'feature1', 'feature2', 'feature3', 'feature4',
                        'upconv1', 'upconv2', 'act', 'rbox', 'angle', 'conv_attenton']

  sp = torch.load(fname) 
  step = sp['step'] if not reset_step else 0
  try:
    learning_rate = sp['learning_rate']
  except:
    import traceback
    traceback.print_exc()
    learning_rate = 0.001
  opt_state = sp['optimizer']
  sp = sp['state_dict']

  for k, v in net.state_dict().items():
    try:
      if ((load_shared and any(substring in k for substring in shared_layers)) or
            (load_ocr and any(substring in k for substring in ocr_layers)) or
            (load_detection and any(substring in k for substring in detection_layers))):
        param = sp[k]
        v.copy_(param)
    except:
      import traceback
      traceback.print_exc()
  
  if optimizer is not None and load_optimizer:  
    try:
      optimizer.load_state_dict(opt_state)
    except:
      import traceback
      traceback.print_exc()
  
  print(fname)
  return step, learning_rate 
  
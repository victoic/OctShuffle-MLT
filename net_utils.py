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

def freeze_shared(net):
  for params in net.layer0.parameters():
    params.requires_grad = False
  for params in net.layer0_1.parameters():
    params.requires_grad = False

def freeze_detection(net):
  for params in net.layer1.parameters():
    params.requires_grad = False
  for params in net.layer2.parameters():
    params.requires_grad = False
  for params in net.layer3.parameters():
    params.requires_grad = False
  for params in net.layer4.parameters():
    params.requires_grad = False
  for params in net.feature1.parameters():
    params.requires_grad = False
  for params in net.feature2.parameters():
    params.requires_grad = False
  for params in net.feature3.parameters():
    params.requires_grad = False
  for params in net.feature4.parameters():
    params.requires_grad = False
  for params in net.upconv1.parameters():
    params.requires_grad = False
  for params in net.upconv2.parameters():
    params.requires_grad = False
  for params in net.act.parameters():
    params.requires_grad = False
  for params in net.rbox.parameters():
    params.requires_grad = False
  for params in net.angle.parameters():
    params.requires_grad = False
  for params in net.conv_attenton.parameters():
    params.requires_grad = False

def freeze_ocr(net):
  for params in net.conv5.parameters():
    params.requires_grad = False
  for params in net.conv6.parameters():
    params.requires_grad = False
  for params in net.conv7.parameters():
    params.requires_grad = False
  for params in net.conv8.parameters():
    params.requires_grad = False
  for params in net.conv9_1.parameters():
    params.requires_grad = False
  for params in net.conv9_2.parameters():
    params.requires_grad = False
  for params in net.conv10_s.parameters():
    params.requires_grad = False
  for params in net.conv11.parameters():
    params.requires_grad = False
  for params in net.batch5.parameters():
    params.requires_grad = False
  for params in net.batch6.parameters():
    params.requires_grad = False
  for params in net.batch7.parameters():
    params.requires_grad = False
  for params in net.batch8.parameters():
    params.requires_grad = False
  for params in net.batch9.parameters():
    params.requires_grad = False
  for params in net.batch10_s.parameters():
    params.requires_grad = False
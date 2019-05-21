from thop import profile, count_hooks
import torch.nn as nn
import torch
import models

def count_Conv2d(m, x, y):
  x = x[0]
  cin = m.in_channels
  # batch_size = x.size(0)

  kernel_ops = torch.Tensor(m.weight.size()[2:]).numel()
  bias_ops = 1 if m.bias is not None else 0
  ops_per_element = kernel_ops + bias_ops
  output_elements = y.nelement()

  # cout x oW x oH
  total_ops = cin * output_elements * ops_per_element // m.groups
  m.total_ops = torch.Tensor([int(total_ops)])

def count_Cat(m, x, y):
  if isinstance(x, tuple):
    total_ops = 2
  else:
    total_ops = 1
  m.total_ops = torch.Tensor([int(total_ops)])

def count_leakyReLU(m, x, y):
  x = x[0]
  hf, lf = x

  nelements_hf = hf.numel()
  nelements_lf = lf.numel()
  total_ops = nelements_hf + nelements_lf

  m.total_ops = torch.Tensor([int(total_ops)])
    
def profile_model():
  model = models.ModelResNetSep2()
  flops, _ = profile(
    model,
    input_size=(1, 3, 224,224),
    custom_ops={
      nn.InstanceNorm2d: count_hooks.count_bn,
      models._Cat: count_Cat,
      models._LeakyReLU: count_leakyReLU,
      nn.Dropout2d: None,
      nn.MSELoss: None,
      nn.SmoothL1Loss: None,
      nn.Conv2d: count_Conv2d
    }
  )
  print("FLOPS: {}".format(flops))

if __name__ == '__main__': 
  profile_model()
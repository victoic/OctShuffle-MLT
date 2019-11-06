import glob, os
from models import OctMLT
from ocr_test_utils import print_seq_ext, test
from datetime import date

def test_stages(opts)
  net = OctMLT(attention=True)
  if opts.cuda:
    net.cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)

  save_file = date.today().strftime("%b-%d-%Y")+".csv"
  save_path = os.path.join(opts.results_dir, save_file)
  result_file = open(save_path, "w")
  result_file.write("model,acc_val,ted\n")

  os.chdir(opts.dir)
  for stage in glob.glob("*.h5"):
    name = stage.split('.')[0]
    if os.path.exists(stage):
      print('loading model from %s' % args.model)
      step_start, learning_rate = net_utils.load_net(stage, net, optimizer,
                                  load_ocr=1, 
                                  load_detection=0, 
                                  load_shared=1,
                                  load_optimizer=1,
                                  reset_step=0)
    net_utils.freeze_shared(net)
    net_utils.freeze_ocr(net)
    net_utils.freeze_detection(net)
    print('starting test..')
    acc_val, ted = test(net, opts.codec, opts, list_file=opts.valid_list, norm_height=opts.norm_height)
    print('saving results..')
    result_file.write("{name},{acc_val},{ted}\n")
  text_file.close()
  print('end')

if __name__ == '__main__': 
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-dir', default='/backup2')
  parser.add_argument('-codec', default='codec.txt')
  parser.add_argument('-valid_list', default='/dataset/VALIDATION/ICDAR2017MLT/crops_list.txt')
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-norm_height', type=int, default=40)
  parser.add_argument('-results_dir', default='/test_results')
  
  args = parser.parse_args()
  test_stages(args)
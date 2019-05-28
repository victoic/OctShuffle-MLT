import argparse
import os
def generate_crops_list(path = "./dataset/crops/"):
    list_file = open(path+"crops_list.txt", "w")
    for r, d, f in os.walk(path):
        for file in f:
            if 'gt.txt' in file:
                rel_dir = os.path.relpath(r, path)
                txt = open(os.path.join(path,rel_dir,file), "r")
                for line in txt:
                    list_file.write('%s/%s' % (rel_dir, line))
                txt.close()
    list_file.close()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-path', default="./dataset/crops/")
  args = parser.parse_args()
  generate_crops_list(args.path)
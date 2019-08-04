import argparse
import os
def make_list(images_path, txt_name):
  path = images_path
  files = []
  # r=root, d=directories, f = files
  for r, d, f in os.walk(path):
    for file in f:
      if '.txt' not in file:
        rel_dir = os.path.relpath(r, path)
        files.append(os.path.join(rel_dir, file)+'\n')

  txt = open(os.path.join(images_path, txt_name), 'a')
  txt.writelines(files)
  txt.close()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-images_path', default='dataset/images/')
  parser.add_argument('-txt_name', default='trainMLT.txt')

  args = parser.parse_args()
  make_list(args.images_path, args.txt_name)
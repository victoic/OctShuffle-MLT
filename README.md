# OctShuffle-MLT

Repository for the paper OctShuffleMLT: A Compact Octave Based Neural Network for End-to-End Multilingual Text Detection and Recognition.

We use E2E-MLT (https://github.com/MichalBusta/E2E-MLT) as baseline, modifying it to obtain a compacter model.

To use the more robust and less compact model Oct-MLT, check the octmlt branch.

## Requirements

- python3.x (3.6 used)
- opencv-python
- pytorch 0.4.1
- torchvision 
- torch_baidu_ctc (https://pypi.org/project/torch-baidu-ctc/0.1.1/)

## Data

Similiar to E2E-MLT we use the following datasets

- ICDAR 2019 MLT Dataset
- ICDAR 2017 MLT Dataset
- ICDAR 2015 Dataset
- RCTW-17
- Synthetic MLT Data (Arabic, Bangla, Chinese, Japanese, Korean, Latin, Hindi )
- and converted GT to icdar MLT format (see: http://rrc.cvc.uab.es/?ch=8&com=tasks) (Arabic, Bangla, Chinese, Japanese, Korean, Latin, Hindi )
    
## Training
    
Use the train.py script to start training. It has the following arguments:
    
- -train_list: Text file with list of images for detection to be trained upon. Default='dataset/images/trainMLT.txt'
- -ocr_feed_list: Text file with list of images for recognition to be trained upon. Default='dataset/crops/crops_list.txt'
- -save_path: Path to save model on checkpoints. Default='backup'
- -model: Model to load on training, if not set training starts from 0. Default=''
- -debug: Prints some informations during training. Default=0
- -batch_size: Batch size for detection training. Default=32
- -ocr_batch_size: Batch size for recognition training. Default=256
- -num_readers: Number of readers. Default=1
- -cuda: Sets use of GPU. Default=True
- -input_size: Input image size. Default=256
- -base_lr: Base Learning Rate for the Adam Optmizer. Default=0.0001
- -max_iters: Maximum number of training iterations. Default=300000

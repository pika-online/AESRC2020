#!/bin/sh
#set -e




 python -u train_aesrc.py res34 0 bigru softmax 0.0 0 0
 python -u train_aesrc.py res34 1 bigru softmax 0.0 0 0
# python train_aesrc.py res34 1 avg softmax 0.0 0 /disc1/ARNet/exp/libri/asr_bpe1000/016.h5

#python -u train_aesrc.py res34 0 bigru softmax 0.0 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru softmax 0.0 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
#python -u train_aesrc.py res34 1 bigru cosface 0.1 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.3 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
#python -u train_aesrc.py res34 1 bigru arcface 0.1 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.3 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
#python -u train_aesrc.py res34 1 bigru circleloss 0.1 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru circleloss 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru circleloss 0.3 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
python -u train_aesrc.py res34 1 avg circleloss 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
python -u train_aesrc.py res34 1 vlad circleloss 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
python -u train_aesrc.py res34 1 gvlad circleloss 0.2 0 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5


#python -u train_aesrc.py res34 1 bigru softmax 0.0 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.0 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.1 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.2 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.3 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
#python -u train_aesrc.py res34 1 bigru arcface 0.0 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.1 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.2 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.3 2 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#
#python -u train_aesrc.py res34 1 bigru softmax 0.0 3 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru cosface 0.2 3 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru arcface 0.3 3 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
#python -u train_aesrc.py res34 1 bigru circleloss 0.2 3 /disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=avg_loss=softmax_magrin=0.00_bn=0_init=1/012.h5
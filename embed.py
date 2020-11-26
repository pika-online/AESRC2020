import local.model as mdl
import local.utils as us
import numpy as np
import os
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_INPUT_LEN = 1200
MAX_LABEL_LEN = 72
ENCODER_LEN = 114
BATCH_SIZE = 180
INIT_EPOCH = 0
FEAT_DIM = 80
BPE_CLASSES = 1000
ACCENT_CLASSES = 8
EPOCHS = 15
ENCODER_RNN_NUM = 1
ASR_RNN_NUM = 1
HIDDEN_DIM = 256
CTC_WT = 0.4
MT_WT = 0.6
AR_WT = 0.01
RES_TYPE = 'res34'
AGGREGATION = 'bigru'
METRIC_LOSS = "circleloss"
MARGIN=0.2
BN_DIM = 3
task = "%s_%s"%(AGGREGATION,METRIC_LOSS)
RAW_MODEL = "/disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=bigru_loss=circleloss_magrin=0.20_bn=3_init=1/007.h5"


# file
train_file = "/disc1/AESRC2020/data/aesrc_fbank/demo.scp"
dev_file = "/disc1/AESRC2020/data/aesrc_fbank/dev.scp"

# feats
FEATS = us.kaldiio.load_scp(train_file)
FEATS_DEV = us.kaldiio.load_scp(dev_file)

train_lst = us.scp2key(us.read_lines(train_file))
dev_lst = us.scp2key(us.read_lines(dev_file))

train_data = us.load_sarnet(train_lst, FEATS,
                              encoder_len=ENCODER_LEN,
                              max_input_len=MAX_INPUT_LEN,
                              max_label_len=MAX_LABEL_LEN,
                              trans_ids=us.AESRC_TRANS_IDS_1K,
                              accent_classes=ACCENT_CLASSES,
                              accent_dct=us.AESRC_ACCENT,
                              accent_ids=us.AESRC_ACCENT2INT)

dev_data = us.load_sarnet(dev_lst, FEATS_DEV,
                              encoder_len=ENCODER_LEN,
                              max_input_len=MAX_INPUT_LEN,
                              max_label_len=MAX_LABEL_LEN,
                              trans_ids=us.AESRC_TRANS_IDS_1K,
                              accent_classes=ACCENT_CLASSES,
                              accent_dct=us.AESRC_ACCENT,
                              accent_ids=us.AESRC_ACCENT2INT)



model =  mdl.SAR_Net(input_shape=[MAX_INPUT_LEN, FEAT_DIM, 1],
                        asr_enable=True,
                        ar_enable=True,
                        res_type=RES_TYPE,
                        res_filters=32,
                        hidden_dim=HIDDEN_DIM,
                        bn_dim=BN_DIM,
                        encoder_rnn_num=ENCODER_RNN_NUM,
                        asr_rnn_num=ASR_RNN_NUM,
                        bpe_classes=BPE_CLASSES,
                        acc_classes=ACCENT_CLASSES,
                        max_label_len=MAX_LABEL_LEN,
                        mto=AGGREGATION,
                        metric_loss=METRIC_LOSS,
                        margin=MARGIN,
                        raw_model=RAW_MODEL,
                        name=None)


inputs = model.get_layer(name="inputs").input
outputs1 = model.get_layer(name="bottleneck").output
sub_model = Model(inputs=[inputs],outputs=[outputs1])


bn = sub_model.predict(x=train_data[0], batch_size=256)
us.save('embedding/train_%s_%s_%.2f_%dD.pkl' % (AGGREGATION,METRIC_LOSS,MARGIN,BN_DIM),
        {"bn":bn,
         'labels':np.argmax(train_data[1]['accent_labels'], axis=1)})

bn = sub_model.predict(x=dev_data[0], batch_size=256)
us.save('embedding/dev_%s_%s_%.2f_%dD.pkl' % (AGGREGATION,METRIC_LOSS,MARGIN,BN_DIM),
        {"bn":bn,
         'labels':np.argmax(dev_data[1]['accent_labels'], axis=1)})
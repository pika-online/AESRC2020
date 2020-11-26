import local.utils as us
import local.model as mdl
from keras.models import load_model
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dev_file = "/disc1/AESRC2020/data/aesrc_fbank/dev.scp"
test_file = "/disc1/AESRC2020/data/aesrc_test/feats.scp"
dev_lst = us.scp2key(us.read_lines(dev_file))
test_lst = us.scp2key(us.read_lines(test_file))
FEATS = us.kaldiio.load_scp('/disc1/AESRC2020/data/aesrc.feats.scp')

dev_inputs = []
test_inputs = []
for key in dev_lst:dev_inputs += [us.feat_reshape(us.feat_norm(FEATS[key]),1200)]
for key in test_lst:test_inputs += [us.feat_reshape(us.feat_norm(FEATS[key]),1200)]
dev_inputs = np.float32(np.expand_dims(np.asarray(dev_inputs), axis=3))
test_inputs = np.float32(np.expand_dims(np.asarray(test_inputs), axis=3))

dev_labs = []
test_labs = []
for key in dev_lst:dev_labs += [us.to_categorical(us.AESRC_ACCENT2INT[us.AESRC_ACCENT[key]],8)]
for key in test_lst:test_labs += [us.to_categorical(us.AESRC_ACCENT2INT[us.AESRC_ACCENT[key]],8)]


def acc(pred, labs):
    print("Overall", us.accent_acc(labs, pred))
    print("Chinese", us.accent_acc(labs, pred, 0))
    print("Japanese", us.accent_acc(labs, pred, 1))
    print("Indian", us.accent_acc(labs, pred, 2))
    print("Korea", us.accent_acc(labs, pred, 3))
    print("American", us.accent_acc(labs, pred, 4))
    print("Britain", us.accent_acc(labs, pred, 5))
    print("Portuguese", us.accent_acc(labs, pred, 6))
    print("Russian", us.accent_acc(labs, pred, 7))


model = mdl.SAR_Net(input_shape=[1200, 80, 1],
                    ar_enable=True,
                    asr_enable=True,
                    res_type="res34",
                    res_filters=32,
                    hidden_dim=256,
                    bn_dim=0,
                    encoder_rnn_num=1,
                    asr_rnn_num=1,
                    bpe_classes=1000,
                    acc_classes=8,
                    max_label_len=72,
                    mto='gvlad',
                    vlad_clusters=8,
                    ghost_clusters=2,
                    metric_loss='circleloss',
                    margin=0.2,
                    raw_model="/disc1/ARNet/exp/aesrc/cnn=res34_asr=1_integration=gvlad_loss=circleloss_magrin=0.20_bn=0_init=1/004.h5",
                    name=None)

accent_model = mdl.sub_model(model, 'inputs', 'accent_labels')


dev_pred = accent_model.predict(dev_inputs, batch_size=300)
test_pred = accent_model.predict(test_inputs, batch_size=300)


acc(dev_pred,dev_labs)
acc(test_pred,test_labs)

exit()
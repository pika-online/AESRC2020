import local.utils as us
import local.model as mdl
import os
import random
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback,ModelCheckpoint,LearningRateScheduler
import local.losses as ls
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys



def train():

    model = mdl.SAR_Net(input_shape=[MAX_INPUT_LEN, FEAT_DIM, 1],
                        asr_enable=ASR_EN,
                        ar_enable=True,
                        res_type=RES_TYPE,
                        res_filters=RES_FILTERS,
                        hidden_dim=HIDDEN_DIM,
                        bn_dim=BN_DIM,
                        encoder_rnn_num=ENCODER_RNN_NUM,
                        asr_rnn_num=ASR_RNN_NUM,
                        bpe_classes=BPE_CLASSES,
                        acc_classes=ACC_CLASSES,
                        max_label_len=MAX_LABEL_LEN,
                        mto=MANY_TO_ONE,
                        metric_loss=METRIC_LOSS,
                        margin=MARGIN,
                        raw_model=RAW_MODEL,
                        name=None)

    train_model = mdl.compile(model,gpus=2,lr=0.001,
                                 loss=loss,
                                 loss_weights=loss_weights,
                                 metrics=metrics)



    class evaluation(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.device("/cpu:0"):
                print("============== SAVING =============")
                model.save("%s/%03d.h5" % (MODEL_DIR, epoch))

    EVL = evaluation()
    #
    train_model.fit_generator(generator=generator, steps_per_epoch=N_BATCHS, epochs=EPOCHS,
                                 callbacks=[ early_stopper,lr_reducer,csv_logger, EVL], initial_epoch=INIT_EPOCH,
                                 validation_data=(dev_data[0], dev_data[1]),max_queue_size=20)







if __name__ == "__main__":

    # network
    MAX_INPUT_LEN = 1200
    FEAT_DIM = 80
    MAX_LABEL_LEN = 72
    # RES_TYPE = 'res34'
    RES_FILTERS = 32
    HIDDEN_DIM = 256
    ENCODER_RNN_NUM = 1
    ASR_RNN_NUM = 1
    AR_RNN_NUM = 1
    BPE_CLASSES = 1000
    ACCENT_CLASSES = 8
    ACC_CLASSES = 8
    # MANY_TO_ONE = "avg"
    # METRIC_LOSS = "softmax"
    # MARGIN = "0.00"

    # training
    ENCODER_LEN = 114
    BATCH_SIZE = 150
    INIT_EPOCH = 0
    EPOCHS = 30
    MT_WT = 0.6
    AR_WT = 0.01
    if BPE_CLASSES == 1000:
        BPE = us.BPE_EN_1K
        TRANS_IDS = us.AESRC_TRANS_IDS_1K
    else:
        BPE = us.BPE_EN_3K
        TRANS_IDS = us.AESRC_TRANS_IDS_3K

    # Hyper-Params
    if len(sys.argv)!=8:
        print('Usage: ',
              'python train_aesrc.py <CNNs> <ASR_EN> <Integration> <Disc Loss> <Margin> <BN> <Init>',
              'Note: ',
              "<CNNs>: res18, res34, res50, res101, res152",
              "<ASR_EN>: enable ASR branch (>0)",
              "<Integration>: avg, bigru, attention, vlad, gvlad",
              "<Disc Loss>: softmax, sphereface, cosface, arcface, circleloss",
              "<Margin>: margin for loss",
              "<init>: initial model")
    RES_TYPE = sys.argv[1]
    ASR_EN = int(sys.argv[2])
    MANY_TO_ONE = sys.argv[3]
    METRIC_LOSS = sys.argv[4]
    MARGIN = float(sys.argv[5])
    BN_DIM = int(sys.argv[6])
    RAW_MODEL = sys.argv[7] if os.path.isfile(sys.argv[7]) else None

    task = "cnn=%s_asr=%d_integration=%s_loss=%s_magrin=%.2f_bn=%d_init=%d"\
           %(RES_TYPE, ASR_EN, MANY_TO_ONE, METRIC_LOSS, MARGIN,BN_DIM,1 if RAW_MODEL else 0)
    MODEL_DIR = "exp/aesrc/%s/" % task
    if not os.path.isdir(MODEL_DIR): os.mkdir(MODEL_DIR)



    # file
    train_file = "/disc1/AESRC2020/data/aesrc_fbank/train.scp"
    dev_file = "/disc1/AESRC2020/data/aesrc_fbank/dev.scp"

    # feats
    FEATS = us.kaldiio.load_scp(train_file)
    FEATS_DEV = us.kaldiio.load_scp(dev_file)

    # list
    train_lst = us.scp2key(us.read_lines(train_file))
    dev_lst = us.scp2key(us.read_lines(dev_file))
    N_BATCHS = len(train_lst) // BATCH_SIZE

    lr_reducer = ReduceLROnPlateau(factor=0.3, cooldown=0, patience=1, min_lr=1e-5,
                                   monitor='val_accent_labels_acc', mode='max', min_delta=0.001, verbose=1)
    early_stopper = EarlyStopping(patience=3,
                                  monitor='val_accent_labels_acc', mode='max', min_delta=0.001, verbose=1)
    csv_logger = CSVLogger('%s/train.csv' % MODEL_DIR)


    # generator
    generator = us.generator_sarnet(train_lst, FEATS, BATCH_SIZE,
                                    encoder_len=ENCODER_LEN,
                                    max_input_len=MAX_INPUT_LEN,
                                    max_label_len=MAX_LABEL_LEN,
                                    trans_ids=TRANS_IDS,
                                    accent_classes=ACCENT_CLASSES,
                                    accent_dct=us.AESRC_ACCENT,
                                    accent_ids=us.AESRC_ACCENT2INT)

    dev_data = us.load_sarnet(dev_lst, FEATS_DEV,
                              encoder_len=ENCODER_LEN,
                              max_input_len=MAX_INPUT_LEN,
                              max_label_len=MAX_LABEL_LEN,
                              trans_ids=TRANS_IDS,
                              accent_classes=ACCENT_CLASSES,
                              accent_dct=us.AESRC_ACCENT,
                              accent_ids=us.AESRC_ACCENT2INT)



    # loss
    loss = {"accent_metric": 'categorical_crossentropy' if METRIC_LOSS!='circleloss'\
                                                        else lambda y,x:ls.circle_loss(y,x,gamma=256,margin=MARGIN),
            "accent_labels": 'categorical_crossentropy'}
    loss_weights = {"accent_metric": MT_WT if ASR_EN else 1.0,
                    "accent_labels": AR_WT}
    metrics = {"accent_metric": 'accuracy',
                "accent_labels": 'accuracy'}
    if ASR_EN:
        loss["ctc_loss"] = lambda y_true, y_pred: y_pred
        loss_weights["ctc_loss"] = 1-MT_WT

    if BN_DIM:
        loss["accent_bn"] = 'categorical_crossentropy' if METRIC_LOSS!='circleloss'\
                                                        else lambda y,x:ls.circle_loss(y,x,gamma=256,margin=MARGIN)
        metrics['accent_bn'] = 'accuracy'
        loss_weights["accent_bn"] = 0.1


    train()
    exit()



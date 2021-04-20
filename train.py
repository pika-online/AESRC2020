import utils as us
import model as mdl
import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    model,train_model = mdl.SAR_Net(input_shape=[MAX_INPUT_LEN,FEAT_DIM,1],
                                    ctc_enable = CTC_EN,
                                    ar_enable = AR_EN,
                                    disc_enable = DISC_EN,
                                    res_type=RES_TYPE,
                                    res_filters=RES_FILTERS,
                                    hidden_dim=HIDDEN_DIM,
                                    bn_dim=BN_DIM,
                                    bpe_classes=BPE_CLASSES,
                                    accent_classes=ACCENT_CLASSES,
                                    max_ctc_len=MAX_CTC_LEN,
                                    mto=MANY_TO_ONE,
                                    vlad_clusters=8,
                                    ghost_clusters=2,
                                    metric_loss=METRIC_LOSS,
                                    margin=MARGIN,
                                    raw_model=RAW_MODEL,
                                    lr=LR,
                                    gpus = GPUS,
                                    name=None)

    class evaluation(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.device("/cpu:0"):
                print("============== SAVING =============")
                model.save("%s/%03d.h5" % (MODEL_DIR, epoch))

    EVL = evaluation()
    train_model.fit_generator(generator=train_generator,
                              steps_per_epoch=N_BATCHS,
                              epochs=EPOCHS,
                              callbacks=[early_stopper,lr_reducer,csv_logger, EVL],
                              initial_epoch=INIT_EPOCH,
                              validation_data=(dev_data[0], dev_data[1]),
                              max_queue_size=20)


if __name__ == "__main__":

    # network
    MAX_INPUT_LEN = 1200
    FEAT_DIM = 80
    MAX_CTC_LEN = 72
    RES_TYPE = 'res34'
    RES_FILTERS = 32
    HIDDEN_DIM = 256
    BPE_CLASSES = 1000
    ACCENT_CLASSES = 8
    MANY_TO_ONE = "bigru"
    METRIC_LOSS = "softmax"
    MARGIN = 0.3
    BN_DIM = 0

    # ENABLE
    AR_EN = True
    CTC_EN = True
    DISC_EN = True

    # training
    ENCODER_LEN = 114
    BATCH_SIZE = 32
    INIT_EPOCH = 0
    EPOCHS = 30
    LR = 0.01
    GPUS = 1
    RAW_MODEL = None
    TASK = "CNN@%s_AR@%d_DISC@%d_CTC@%d_MTO@%s_DLOSS@%s_MARGIN@%.2f_INIT@%d_BN@%d"%(RES_TYPE,AR_EN,DISC_EN,CTC_EN,MANY_TO_ONE,
                                                                      METRIC_LOSS,MARGIN,1 if RAW_MODEL else 0,BN_DIM)
    MODEL_DIR = "exp/%s"%TASK
    if not os.path.isdir(MODEL_DIR):os.mkdir(MODEL_DIR)
    lr_reducer = ReduceLROnPlateau(factor=0.3, cooldown=0, patience=2, min_lr=1e-5,
                                   monitor='val_y_accent_acc', mode='max', min_delta=0.001, verbose=1)
    early_stopper = EarlyStopping(patience=3,
                                  monitor='val_y_accent_acc', mode='max', min_delta=0.001, verbose=1)
    csv_logger = CSVLogger('%s/train.csv' % MODEL_DIR)


    # generator and data
    DATA_DCT = us.load("array/data_scp.pkl")
    ACCENT_DCT = us.load("array/accent_scp.pkl")
    TRANS_DCT = us.load("array/trans_scp.pkl")
    train_lst = us.read_lines("train.lst")
    dev_lst = us.read_lines("dev.lst")[:100]
    N_BATCHS = len(train_lst)//BATCH_SIZE

    train_generator = us.data_generator(train_lst,
                                  batch_size=BATCH_SIZE,
                                  ctc_enable=CTC_EN,
                                  ar_enable=AR_EN,
                                  disc_enable=DISC_EN,
                                  data_dct=DATA_DCT,
                                  accent_dct=ACCENT_DCT,
                                  trans_dct=TRANS_DCT,
                                  max_input_len=MAX_INPUT_LEN,
                                  max_ctc_len=MAX_CTC_LEN,
                                  encoder_len=ENCODER_LEN,
                                  accent_classes=ACCENT_CLASSES,)

    dev_data = us.data_loader(dev_lst,
                              ctc_enable=CTC_EN,
                              ar_enable=AR_EN,
                              disc_enable=DISC_EN,
                              data_dct=DATA_DCT,
                              accent_dct=ACCENT_DCT,
                              trans_dct=TRANS_DCT,
                              max_input_len=MAX_INPUT_LEN,
                              max_ctc_len=MAX_CTC_LEN,
                              encoder_len=ENCODER_LEN,
                              accent_classes=ACCENT_CLASSES,)


    train()
    exit()



import local.utils as us
import local.model as mdl
import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback,ModelCheckpoint,LearningRateScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train():

    model = mdl.SAR_Net(input_shape=[MAX_INPUT_LEN, FEAT_DIM, 1],
                        asr_enable=True,
                        ar_enable=False,
                        res_type=RES_TYPE,
                        res_filters=RES_FILTERS,
                        hidden_dim=HIDDEN_DIM,
                        bn_dim=0,
                        encoder_rnn_num=ENCODER_RNN_NUM,
                        asr_rnn_num=ASR_RNN_NUM,
                        bpe_classes=BPE_CLASSES,
                        acc_classes=ACC_CLASSES,
                        max_label_len=MAX_LABEL_LEN,
                        mto=MANY_TO_ONE,
                        metric_loss=METRIC_LOSS,
                        margin=MARGIN,
                        raw_model=RAW_MODEL,
                        name=task)

    train_model = mdl.compile(model,gpus=2,lr=0.001,
                                 loss={"ctc_loss": lambda y_true, y_pred: y_pred},
                                 loss_weights={"ctc_loss": 1.0},
                                 metrics={})

    with tf.device("/cpu:0"):
        ctc_decode_model = mdl.sub_model(model, 'inputs', 'ctc_pred')

    class evaluation(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.device("/cpu:0"):
                print("============== SAVING =============")
                model.save_weights("%s/%03d.h5" % (MODEL_DIR, epoch))

                print("============ ASR EVAL ==========")
                ctc_pred = mdl.ctc_pred(ctc_decode_model, dev_data[0], input_len=ENCODER_LEN,batch_size=BATCH_SIZE)
                print("DEV-WER:", us.wer_eval(dev_data[0]["ctc_labels"],
                                              dev_data[0]["ctc_label_len"],
                                              ctc_pred, bpe=BPE, show=True))

    EVL = evaluation()
    #
    train_model.fit_generator(generator=generator, steps_per_epoch=N_BATCHS, epochs=EPOCHS,
                                 callbacks=[ early_stopper,lr_reducer,csv_logger, EVL], initial_epoch=INIT_EPOCH,
                                 validation_data=(dev_data[0], dev_data[1]),max_queue_size=20)

if __name__ == "__main__":

    # network
    MAX_INPUT_LEN = 1800
    FEAT_DIM = 80
    MAX_LABEL_LEN = 120
    RES_TYPE = 'res34'
    RES_FILTERS = 32
    HIDDEN_DIM = 256
    ENCODER_RNN_NUM = 1
    ASR_RNN_NUM = 1
    AR_RNN_NUM = 1
    BPE_CLASSES = 1000
    ACC_CLASSES = 8
    MANY_TO_ONE = None
    METRIC_LOSS = "softmax"
    MARGIN = "0.00"

    # training
    ENCODER_LEN = 171
    BATCH_SIZE = 100
    INIT_EPOCH = 8
    EPOCHS = 30

    task = "asr_bpe%d"%BPE_CLASSES
    RAW_MODEL = '/disc1/ARNet/exp/libri/asr_bpe1000/008.h5'
    MODEL_DIR = "exp/libri/%s/" % task
    LOG_FILE = "%s/train.csv"%MODEL_DIR
    if not os.path.isdir(MODEL_DIR): os.mkdir(MODEL_DIR)

    if BPE_CLASSES == 1000:
        BPE = us.BPE_EN_1K
        TRANS_IDS = us.LIBRI_TRANS_IDS_1K
    else:
        BPE = us.BPE_EN_3K
        TRANS_IDS = us.LIBRI_TRANS_IDS_3K


    # file
    train_file = "/disc1/AESRC2020/data/librispeech_train/feats.scp"
    dev_clean_file = "/disc1/AESRC2020/data/librispeech_dev_clean/feats.scp"
    # dev_other_file = "/disc1/AESRC2020/data/librispeech_dev_other/feats.scp"
    # test_clean_file = "/disc1/AESRC2020/data/librispeech_test_clean/feats.scp"
    # test_other_file = "/disc1/AESRC2020/data/librispeech_test_other/feats.scp"

    # feats
    FEATS = us.kaldiio.load_scp(train_file)
    FEATS_DEV = us.kaldiio.load_scp(dev_clean_file)

    # list
    train_lst = us.scp2key(us.read_lines(train_file))
    train_lst = us.limit_samples(train_lst,us.LIBRI_UTT2FRAMES,TRANS_IDS,MAX_INPUT_LEN,MAX_LABEL_LEN)
    dev_lst = us.scp2key(us.read_lines(dev_clean_file))
    dev_lst = us.limit_samples(dev_lst, us.LIBRI_UTT2FRAMES, TRANS_IDS, MAX_INPUT_LEN, MAX_LABEL_LEN)
    N_BATCHS = len(train_lst) // BATCH_SIZE


    # callbacks
    lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=0, min_lr=5e-5,
                                   monitor='val_loss',min_delta=0.5,verbose=1)
    early_stopper = EarlyStopping( min_delta=0.1, patience=5,monitor='val_loss',verbose=1)
    csv_logger = CSVLogger(LOG_FILE)


    # generator
    generator = us.generator_ctc(lst=train_lst,
                                 feats=FEATS,
                                 batch_size=BATCH_SIZE,
                                 max_input_len=MAX_INPUT_LEN,
                                 max_label_len=MAX_LABEL_LEN,
                                 encoder_len=ENCODER_LEN,
                                 trans_ids=TRANS_IDS)

    dev_data = us.load_ctc(lst=dev_lst,
                           feats=FEATS_DEV,
                           max_input_len=MAX_INPUT_LEN,
                           max_label_len=MAX_LABEL_LEN,
                           encoder_len=ENCODER_LEN,
                           trans_ids=TRANS_IDS)



    train()
    exit()



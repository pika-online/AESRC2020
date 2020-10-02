import local.utils as us
import local.model as mdl
import os
import random
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback,ModelCheckpoint,LearningRateScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



def train():

    # RES + RNN
    model = mdl.model_res_gru_ctc(shapes=(MAX_INPUT_LEN, FEAT_DIM, 1),
                                  bpe_classes=BPE_CLASSES,
                                  max_label_len=MAX_LABEL_LEN,
                                  cnn=CNN,
                                  raw_model=RAW_MODEL)

    parallel_model = mdl.compile(model,gpus=3,lr=0.001,
                                 loss={"ctc_loss": lambda y_true, y_pred: y_pred},
                                 loss_weights={"ctc_loss": 1.0},
                                 metrics=None)

    with tf.device("/cpu:0"):
        ctc_decode_model = mdl.sub_model(model, 'inputs', 'ctc_pred')

    class evaluation(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.device("/cpu:0"):
                print("============== SAVING =============")
                model.save_weights("%s/%03d.h5" % (MODEL_DIR, epoch))

                print("============ ATT EVAL ==========")
                ctc_pred = mdl.ctc_pred(ctc_decode_model, dev_data[0], input_len=ENCODER_LEN,
                                        batch_size=BATCH_SIZE)
                print("DEV-WER:", us.ctc_eval(dev_data[0]["ctc_labels"],
                                              dev_data[0]["ctc_label_len"], ctc_pred, True))

    EVL = evaluation()
    #
    parallel_model.fit_generator(generator=generator, steps_per_epoch=N_BATCHS, epochs=EPOCHS,
                                 callbacks=[ early_stopper,lr_reducer,csv_logger, EVL], initial_epoch=INIT_EPOCH,
                                 validation_data=(dev_data[0], dev_data[1]), validation_steps=N_BATCHS)


def test():

    model_file = "/disc1/ARNet/exp/libri/res18_gru/012.h5"

    FEATS_DEV_CLEAN = us.kaldiio.load_scp(dev_clean_file)
    FEATS_DEV_OTHER = us.kaldiio.load_scp(dev_other_file)
    FEATS_TEST_CLEAN = us.kaldiio.load_scp(test_clean_file)
    FEATS_TEST_OTHER = us.kaldiio.load_scp(test_other_file)

    dev_clean_lst = us.scp2key(us.read_lines(dev_clean_file))
    dev_other_lst = us.scp2key(us.read_lines(dev_other_file))
    test_clean_lst = us.scp2key(us.read_lines(test_clean_file))
    test_other_lst = us.scp2key(us.read_lines(test_other_file))

    dev_clean_data = us.load_ctc(dev_clean_lst, FEATS_DEV_CLEAN,
                           encoder_len=ENCODER_LEN,
                           max_input_len=MAX_INPUT_LEN,
                           max_label_len=MAX_LABEL_LEN,
                           trans_ids=us.LIBRI_TRANS_IDS)
    dev_other_data = us.load_ctc(dev_other_lst, FEATS_DEV_OTHER,
                                 encoder_len=ENCODER_LEN,
                                 max_input_len=MAX_INPUT_LEN,
                                 max_label_len=MAX_LABEL_LEN,
                                 trans_ids=us.LIBRI_TRANS_IDS)
    test_clean_data = us.load_ctc(test_clean_lst, FEATS_TEST_CLEAN,
                                 encoder_len=ENCODER_LEN,
                                 max_input_len=MAX_INPUT_LEN,
                                 max_label_len=MAX_LABEL_LEN,
                                 trans_ids=us.LIBRI_TRANS_IDS)
    test_other_data = us.load_ctc(test_other_lst, FEATS_TEST_OTHER,
                                 encoder_len=ENCODER_LEN,
                                 max_input_len=MAX_INPUT_LEN,
                                 max_label_len=MAX_LABEL_LEN,
                                 trans_ids=us.LIBRI_TRANS_IDS)

    model = mdl.model_res_gru_ctc(shapes=(MAX_INPUT_LEN, FEAT_DIM, 1),
                                         bpe_classes=BPE_CLASSES,
                                         max_label_len=MAX_LABEL_LEN,
                                         cnn=CNN,
                                         raw_model=model_file)

    ctc_decode_model = mdl.sub_model(model, 'inputs', 'ctc_pred')
    dev_clean_pred = mdl.ctc_pred(ctc_decode_model, dev_clean_data[0], input_len=ENCODER_LEN, batch_size=BATCH_SIZE)
    dev_other_pred = mdl.ctc_pred(ctc_decode_model, dev_other_data[0], input_len=ENCODER_LEN, batch_size=BATCH_SIZE)
    test_clean_pred = mdl.ctc_pred(ctc_decode_model, test_clean_data[0], input_len=ENCODER_LEN, batch_size=BATCH_SIZE)
    test_other_pred = mdl.ctc_pred(ctc_decode_model, test_other_data[0], input_len=ENCODER_LEN, batch_size=BATCH_SIZE)

    print("dev-clean-wer:",
          us.ctc_eval(dev_clean_data[0]["ctc_labels"],dev_clean_data[0]["ctc_label_len"], dev_clean_pred, True))
    print("dev-other-wer:",
          us.ctc_eval(dev_other_data[0]["ctc_labels"], dev_other_data[0]["ctc_label_len"], dev_other_pred, True))
    print("test-clean-wer:",
          us.ctc_eval(test_clean_data[0]["ctc_labels"], test_clean_data[0]["ctc_label_len"], test_clean_pred, True))
    print("test-other-wer:",
          us.ctc_eval(test_other_data[0]["ctc_labels"], test_other_data[0]["ctc_label_len"], test_other_pred, True))


if __name__ == "__main__":

    # base
    MAX_INPUT_LEN = 1600
    MAX_LABEL_LEN = 100
    ENCODER_LEN = 150
    BATCH_SIZE = 96
    INIT_EPOCH = 0
    FEAT_DIM = 80
    BPE_CLASSES = 1000
    EPOCHS = 20
    CNN = 'res18'
    RAW_MODEL = None
    LOG_FILE = 'exp/libri/res18_gru/model.csv'
    MODEL_DIR = "exp/libri/res18_gru/"

    # file
    train_file = "/disc1/AESRC2020/data/librispeech_train/feats.scp"
    dev_clean_file = "/disc1/AESRC2020/data/librispeech_dev_clean/feats.scp"
    dev_other_file = "/disc1/AESRC2020/data/librispeech_dev_other/feats.scp"
    test_clean_file = "/disc1/AESRC2020/data/librispeech_test_clean/feats.scp"
    test_other_file = "/disc1/AESRC2020/data/librispeech_test_other/feats.scp"

    # feats
    FEATS = us.kaldiio.load_scp(train_file)
    FEATS_DEV = us.kaldiio.load_scp(dev_clean_file)

    # list
    train_lst = us.limit_time_utts(us.limit_trans_utts(us.scp2key(us.read_lines(train_file)),
                                    us.LIBRI_TRANS_IDS, MAX_LABEL_LEN), us.LIBRI_UTT2FRAMES, MAX_INPUT_LEN)
    dev_lst = us.scp2key(us.read_lines(dev_clean_file))
    N_BATCHS = len(train_lst) // BATCH_SIZE


    # callbacks
    lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=0, min_lr=0.5e-6,
                                   monitor='val_loss',min_delta=1.0,verbose=1)
    early_stopper = EarlyStopping( min_delta=0.01, patience=3,monitor='val_loss',verbose=1)
    csv_logger = CSVLogger(LOG_FILE)
    # checkpointer = ModelCheckpoint(filepath=MODEL_FILE, verbose=1, save_best_only=True, monitor='val_loss',
    #                                save_weights_only=True)
    # lrs = LearningRateScheduler(learn_rate,verbose=1)


    # generator
    generator = us.generator_ctc(train_lst, FEATS, BATCH_SIZE,
                                encoder_len=ENCODER_LEN,
                                 max_input_len=MAX_INPUT_LEN,
                                 max_label_len=MAX_LABEL_LEN,
                                 trans_ids=us.LIBRI_TRANS_IDS)

    dev_data = us.load_ctc(dev_lst, FEATS_DEV,
                           encoder_len=ENCODER_LEN,
                           max_input_len=MAX_INPUT_LEN,
                           max_label_len=MAX_LABEL_LEN,
                           trans_ids=us.LIBRI_TRANS_IDS)


    # train()
    test()
    exit()



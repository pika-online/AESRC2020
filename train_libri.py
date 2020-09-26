import local.utils as us
import local.model as mdl
import os
import random
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback,ModelCheckpoint,LearningRateScheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



def train():

    model = mdl.model_ctc( shapes=(MAX_INPUT_LEN, FEAT_DIM, 1),
                                          bpe_classes=BPE_CLASSES,
                                          max_label_len=MAX_LABEL_LEN,
                                          raw_model=RAW_MODEL)

    parallel_model = mdl.compile(model,gpus=3,lr=0.001,
                                 loss=[lambda y_true, y_pred: y_pred],
                                 loss_weights=[1.0],
                                 metrics=None)

    with tf.device("/cpu:0"):
        ctc_decode_model = mdl.sub_model(model, 'inputs', 'ctc_pred')

    class evaluation(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with tf.device("/cpu:0"):
                print("============== SAVING =============")
                model.save_weights("%s/%03d.h5" % (MODEL_DIR, epoch))

                print("============ CTC EVAL ==========")
                ctc_pred = mdl.ctc_pred(ctc_decode_model, dev_data[0], input_len=ENCODER_LEN,
                                        batch_size=BATCH_SIZE)
                print("DEV-WER:", us.ctc_eval(dev_data[0]["ctc_labels"],
                                              dev_data[0]["ctc_label_len"], ctc_pred, True))

    EVL = evaluation()
    #
    parallel_model.fit_generator(generator=generator, steps_per_epoch=N_BATCHS, epochs=EPOCHS,
                                 callbacks=[ early_stopper,lr_reducer,csv_logger, EVL], initial_epoch=INIT_EPOCH,
                                 validation_data=(dev_data[0], dev_data[1]), validation_steps=N_BATCHS)





if __name__ == "__main__":

    # base
    MAX_INPUT_LEN = 1600
    MAX_LABEL_LEN = 100
    ENCODER_LEN = 150
    BATCH_SIZE = 300
    INIT_EPOCH = 2
    FEAT_DIM = 80
    BPE_CLASSES = 1000
    EPOCHS = 30
    RAW_MODEL = '/disc1/ARNet/exp/aesrc/002.h5'
    LOG_FILE = 'exp/libri/model.csv'
    MODEL_DIR = "exp/libri"

    # file
    train_file = "/disc1/AESRC2020/data/librispeech_train/feats.scp"
    dev_file = "/disc1/AESRC2020/data/librispeech_dev_clean/feats.scp"

    # feats
    FEATS = us.kaldiio.load_scp(train_file)
    FEATS_DEV = us.kaldiio.load_scp(dev_file)

    # list
    train_lst = us.limit_time_utts(us.limit_trans_utts(us.scp2key(us.read_lines(train_file)),
                                    us.LIBRI_TRANS_IDS, MAX_LABEL_LEN), us.LIBRI_UTT2FRAMES, MAX_INPUT_LEN)
    dev_lst = us.scp2key(us.read_lines(dev_file))
    N_BATCHS = len(train_lst) // BATCH_SIZE


    # callbacks
    lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=1, min_lr=0.5e-6,
                                   monitor='val_loss',min_delta=0.1,verbose=1)
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


    train()
    exit()



import local.utils as us
import local.model as mdl
import random

dct = {0:"<CHN>",1:"<JPN>",2:"<IND>",3:"<KR>",4:"<US>",5:"<UK>",6:"<PT>",7:"<RU>"}

BPE_CLASSES = 1000
ACCENT_CLASSES = 8
MAX_INPUT_LEN = 1200
ENCODER_LEN = 114
MAX_LABEL_LEN = 72
FEAT_DIM = 80
EPOCHS = 10
BATCH_SIZE = 32
RAW_MODEL = "/disc1/ARNet/exp/aesrc/007.h5"



dev_file = "/disc1/AESRC2020/data/aesrc_fbank_sp/dev.scp"
FEATS_DEV = us.kaldiio.load_scp(dev_file)
dev_lst = us.scp2key(us.read_lines(dev_file))
# dev_lst = random.sample(dev_lst,100)
dev_data = us.load_ctc_accent(dev_lst, FEATS_DEV,
                           encoder_len=ENCODER_LEN,
                           max_input_len=MAX_INPUT_LEN,
                           max_label_len=MAX_LABEL_LEN,
                           trans_ids=us.AESRC_TRANS_IDS,
                           accent_classes=ACCENT_CLASSES,
                           accent_dct=us.AESRC_ACCENT,
                           accent_ids=us.AESRC_ACCENT2INT)





print("==== test ====")
model = mdl.model_ctc_accent( shapes=(MAX_INPUT_LEN, FEAT_DIM, 1),
                                        accent_classes=ACCENT_CLASSES,
                                          bpe_classes=BPE_CLASSES,
                                          max_label_len=MAX_LABEL_LEN,
                                          raw_model=RAW_MODEL)
accent_model = mdl.sub_model(model, 'inputs', 'accent_labels')

accent_pred = accent_model.predict(dev_data[0], batch_size=256)

print("Overall",us.acc(dev_data[1]['accent_labels'], accent_pred))
print("Chinese",us.acc(dev_data[1]['accent_labels'], accent_pred,0))
print("Japanese",us.acc(dev_data[1]['accent_labels'], accent_pred,1))
print("India",us.acc(dev_data[1]['accent_labels'], accent_pred,2))
print("Korea",us.acc(dev_data[1]['accent_labels'], accent_pred,3))
print("American",us.acc(dev_data[1]['accent_labels'], accent_pred,4))
print("Britain",us.acc(dev_data[1]['accent_labels'], accent_pred,5))
print("Portuguese",us.acc(dev_data[1]['accent_labels'], accent_pred,6))
print("Russia",us.acc(dev_data[1]['accent_labels'], accent_pred,7))





# test
# AESRC_FEATS = us.kaldiio.load_scp("data/aesrc_test/feats.scp")
# lst,arr = us.feats2inputs(AESRC_FEATS,max_input_len=1200)
# nation_pred = accent_model.predict({'aesrc_inputs1':arr}, batch_size=256)
# for nation,pred in zip(lst,nation_pred):
#     print(nation,dct[np.argmax(pred)])




exit()
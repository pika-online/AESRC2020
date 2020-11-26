import kaldiio
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from bpemb import BPEmb
from keras.utils.np_utils import to_categorical
import random

"""
================= 
      BASE 
=================
"""
MMS = MinMaxScaler()
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2
BPE_EN_1K = BPEmb(lang="en", vs=1000)
BPE_EN_3K = BPEmb(lang="en", vs=3000)


"""
==========================
        FILE OPT 
==========================
"""

def save(path,arr):
    f = open(path,'wb')
    pickle.dump(arr,f)
    f.close()

def load(path):
    f = open(path,'rb')
    return pickle.load(f)

def read_lines(path):
    f = open(path,'rt')
    lst = [line.strip() for line in f.readlines()]
    f.close()
    return lst

def merge_dct(dict1, dict2):
    res = {**dict1, **dict2}
    return res

"""
==========================
        SCP OPT 
==========================
"""

def scp2key(scp):
    return [line.split()[0] for line in scp]


def path2utt(path):
    return path.split('/')[-1].split('.')[0]


def scp2dct(scp):
    dct = {}
    for line in scp:
        if line.split()[0] in dct:
            print(line)
            break
        dct[line.split()[0]] = " ".join(line.split()[1:])
    return dct



"""
==========================
        FEATS OPT 
==========================
"""

def feat_reshape(feat,max_len=1200):
    h,w = feat.shape
    if h>=max_len:
        return feat[:max_len]
    else:
        feat_ = np.zeros((max_len,w))
        feat_[:h] = feat
        return feat_

def feat_norm(feat):
    return (MMS.fit_transform(np.abs(feat)))


"""
=========================== 
        TEXT OPT 
==========================
"""
def filt_en_word(word):
    return re.sub("[^a-zA-Z^\-^0-9^\']","",word)

def filt_text(text):
    text_ = []
    for word in text.split():
        text_.append(filt_en_word(word))
    return " ".join(text_)

def filt_text_dct(dct):
    for utt in dct:
        dct[utt] = filt_text(dct[utt])
    return dct


def bpe_encoding_ids_with_eos(func,text):
    return func.encode_ids_with_eos(text)

def bpe_decoding_ids(func,ids):
    return func.decode_ids(ids)


def text_ids_norm(ids,max_len):
    if len(ids)>max_len:
        ids = ids[:max_len]
    n = min(len(ids), max_len)
    tmp = [EOS_ID] * max_len
    tmp[:n] = ids
    return tmp


def text_dct_ids_with_eos(func,dct):
    dct_ = {}
    for utt in dct:
        dct_[utt] = bpe_encoding_ids_with_eos(func, dct[utt])
    return dct_


"""
============================
        GENERATOR 
============================
"""
def limit_trans_utts(lst,trans_dct,max_len=64):
    return [utt for utt in lst if len(trans_dct[utt])<=max_len]

def limit_time_utts(lst, utt2frames, max_len=64):
    return [utt for utt in lst if int(utt2frames[utt]) <= max_len]

def limit_samples(lst,utt2frames,trans_dct,max_input_len,max_output_len):
    return limit_trans_utts(limit_time_utts(lst,utt2frames,max_input_len),trans_dct,max_output_len)

# For CTC
def load_ctc(lst, feats,
            max_label_len,
            max_input_len,
            encoder_len,
             trans_ids):
    inputs = []
    ctc_input_len = []
    ctc_label_len = []
    ctc_labels = []
    for utt in lst:
        ctc_label = trans_ids[utt]
        ctc_label_norm = text_ids_norm(ctc_label, max_label_len)
        inputs.append(feat_reshape(feat_norm(feats[utt]),max_input_len))
        ctc_input_len.append(encoder_len)
        ctc_label_len.append(min(len(ctc_label),max_label_len))
        ctc_labels.append(ctc_label_norm)
    return {"inputs": np.float32(np.expand_dims(np.asarray(inputs), axis=3)),
            "ctc_input_len": np.int32(np.expand_dims(np.asarray(ctc_input_len), axis=1)),
            "ctc_label_len": np.int32(np.expand_dims(np.asarray(ctc_label_len), axis=1)),
            "ctc_labels": np.float32(np.asarray(ctc_labels))}, \
           {"ctc_loss": np.zeros([len(lst)])}




def generator_ctc(lst, feats, batch_size,
                  max_label_len,
                  max_input_len,
                  encoder_len,
                 trans_ids
                  ):
    n_batchs = len(lst) // batch_size
    while True:
        random.shuffle(lst)
        for i in range(n_batchs):
            begin = i * batch_size
            end = begin + batch_size
            subs = lst[begin:end]
            data, labels = load_ctc(subs, feats,
                                    max_label_len,
                                    max_input_len,
                                    encoder_len,
                                    trans_ids)
            yield data, labels




def load_sarnet(lst, feats,
                max_label_len,
                max_input_len,
                encoder_len,
                accent_classes,
                trans_ids,
                accent_dct,
                accent_ids, ):
    inputs = []
    ctc_input_len = []
    ctc_label_len = []
    ctc_labels = []
    accent_labels = []
    for utt in lst:
        ctc_label = trans_ids[utt]
        ctc_label_norm = text_ids_norm(ctc_label, max_label_len)
        inputs.append(feat_reshape(feat_norm(feats[utt]),max_input_len))
        ctc_input_len.append(encoder_len)
        ctc_label_len.append(min(len(ctc_label),max_label_len))
        ctc_labels.append(ctc_label_norm)
        accent_labels.append(to_categorical(accent_ids[accent_dct[utt]], num_classes=accent_classes))
    return {"inputs": np.float32(np.expand_dims(np.asarray(inputs), axis=3)),
            "ctc_input_len": np.int32(np.expand_dims(np.asarray(ctc_input_len), axis=1)),
            "ctc_label_len": np.int32(np.expand_dims(np.asarray(ctc_label_len), axis=1)),
            "ctc_labels": np.float32(np.asarray(ctc_labels)),
            "accent_inputs": np.asarray(accent_labels)}, \
           {"ctc_loss": np.zeros([len(lst)]),
            "accent_metric": np.asarray(accent_labels),
            "accent_bn": np.asarray(accent_labels),
            "accent_labels": np.asarray(accent_labels)}




def generator_sarnet(lst, feats, batch_size,
                     max_label_len,
                     max_input_len,
                     encoder_len,
                     accent_classes,
                     trans_ids,
                     accent_dct,
                     accent_ids
                     ):
    n_batchs = len(lst) // batch_size
    while True:
        random.shuffle(lst)
        for i in range(n_batchs):
            begin = i * batch_size
            end = begin + batch_size
            subs = lst[begin:end]
            data, labels = load_sarnet(subs, feats,
                                       max_label_len,
                                       max_input_len,
                                       encoder_len,
                                       accent_classes,
                                       trans_ids,
                                       accent_dct,
                                       accent_ids)
            yield data, labels



"""
============================== 
        METRIC 
==============================
"""
def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1,len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i-1] == word2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def wer_eval(labels, label_lens, preds, bpe, show=None):
    wer_,count = 0,0
    for label,length,pred in zip(labels,label_lens,preds):
        label = [int(item) for item in label]
        length = int(length[0])
        true = label[:length]
        pred = [int(item) for item in pred if item != -1]
        true_ = bpe.decode_ids(true)
        pred_ = bpe.decode_ids(pred) if pred else ""
        wer = edit_distance(true_.split(),pred_.split())/len(true_.split())
        wer_ += wer
        count += 1
    if show:
        print("DEMO-REF", true, true_)
        print("DEMO-HYP", pred, pred_)
    return wer_/count


def accent_acc(ref, hyp, accent=-1):
    ref = np.argmax(ref, axis=1)
    hyp = np.argmax(hyp, axis=1)
    if accent>-1:
        ids = np.where(ref==accent)[0]
        return sum(ref[ids] == hyp[ids])/len(ids)
    else:
        return sum(ref==hyp)/len(ref)

"""
===========================
    PREDEFINATION 
===========================
"""
print("------------ loading predefination ---------------------")
# AESRC_ACCENT = scp2dct(read_lines("/disc1/AESRC2020/src/AESRC2020/NATION.TXT"))
# AESRC_ACCENT2INT = {"Chinese":0, "Japanese":1, "Indian":2, "Korean":3, "American":4, "British":5, "Portuguese":6, "Russian":7}
# AESRC_TRANS = filt_text_dct(scp2dct(read_lines("/disc1/AESRC2020/data/aesrc.trans.scp")))
# AESRC_TRANS_IDS_1K = text_dct_ids_with_eos(BPE_EN_1K, AESRC_TRANS)
# AESRC_TRANS_IDS_3K = text_dct_ids_with_eos(BPE_EN_3K, AESRC_TRANS)
# AESRC_UTT2FRAMES = scp2dct(read_lines("/disc1/AESRC2020/data/aesrc.utt2num_frames"))
#
# LIBRI_TRANS = filt_text_dct(scp2dct(read_lines("/disc1/AESRC2020/data/librispeech.trans.scp")))
# LIBRI_TRANS_IDS_1K = text_dct_ids_with_eos(BPE_EN_1K, LIBRI_TRANS)
# LIBRI_TRANS_IDS_3K = text_dct_ids_with_eos(BPE_EN_3K, LIBRI_TRANS)
# LIBRI_UTT2FRAMES = scp2dct(read_lines("/disc1/AESRC2020/data/librispeech.utt2num_frames"))


# save('/disc1/ARNet/data/dict/aesrc_accent.pkl', AESRC_ACCENT)
# save('/disc1/ARNet/data/dict/aesrc_accent_int.pkl', AESRC_ACCENT2INT)
# save('/disc1/ARNet/data/dict/aesrc_trans_ids_1k.pkl', AESRC_TRANS_IDS_1K)
# save('/disc1/ARNet/data/dict/aesrc_trans_ids_3k.pkl', AESRC_TRANS_IDS_3K)
# save('/disc1/ARNet/data/dict/aesrc_utt2frames.pkl', AESRC_UTT2FRAMES)
#
# save('/disc1/ARNet/data/dict/libri_trans_ids_1k.pkl', LIBRI_TRANS_IDS_1K)
# save('/disc1/ARNet/data/dict/libri_trans_ids_3k.pkl', LIBRI_TRANS_IDS_3K)
# save('/disc1/ARNet/data/dict/libri_utt2frames.pkl',LIBRI_UTT2FRAMES)


AESRC_ACCENT = load('/disc1/ARNet/data/dict/aesrc_accent.pkl')
AESRC_ACCENT2INT = load('/disc1/ARNet/data/dict/aesrc_accent_int.pkl')
AESRC_TRANS_IDS_1K = load('/disc1/ARNet/data/dict/aesrc_trans_ids_1k.pkl')
# AESRC_TRANS_IDS_3K = load('/disc1/ARNet/data/dict/aesrc_trans_ids_3k.pkl')
AESRC_UTT2FRAMES = load('/disc1/ARNet/data/dict/aesrc_utt2frames.pkl')
# LIBRI_TRANS_IDS_1K = load('/disc1/ARNet/data/dict/libri_trans_ids_1k.pkl')
# LIBRI_TRANS_IDS_3K = load('/disc1/ARNet/data/dict/libri_trans_ids_3k.pkl')
# LIBRI_UTT2FRAMES = load('/disc1/ARNet/data/dict/libri_utt2frames.pkl')

if __name__ == "__main__":


    # libri_train = scp2dct(read_lines('/disc1/AESRC2020/data/librispeech_train/feats.scp'))
    # libri_train2 = limit_samples(libri_train, LIBRI_UTT2FRAMES, LIBRI_TRANS_IDS_1K, 1800, 120)



    exit()

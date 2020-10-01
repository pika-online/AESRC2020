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
PUNC = ['-', '.', '~', '\'']
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2
BPE_EN = BPEmb(lang="en", vs=1000)


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

def utt_norm(utt):
    if 'sp' in utt:
        return utt.split('-')[1]
    return utt


"""
==========================
        FEATS OPT 
==========================
"""

def get_cmvn(ark):
    m,v = kaldiio.load_mat(ark)
    m_,n = m[:-1],m[-1]
    v_ = v[:-1]
    return m_/n, v_/n


def feat_reshape(feat,max_len=1200):
    h,w = feat.shape
    if h>=max_len:
        return feat[:max_len]
    else:
        feat_ = np.zeros((max_len,w))
        feat_[:h] = feat
        return feat_

# def feat_norm(feat):
#     return (MMS.fit_transform(np.abs(feat)))

# def feat_norm(feat):
    # eps = 1e-5
    # m,s = np.mean(feat,axis=1,keepdims=True),np.std(feat,axis=1,keepdims=True)
    # return (feat-m)/(s+eps)

def feat_norm(feat):
    # cmn
    m = np.expand_dims(M,axis=0)
    return feat-m



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
        if word in PUNC:continue
        text_.append(filt_en_word(word))
    return " ".join(text_)

def filt_text_dct(dct):
    for utt in dct:
        dct[utt] = filt_text(dct[utt])
    return dct

def bpe_encoding(func,text):
    return func.encode(text)

def bpe_encoding_ids(func,text):
    return func.encode_ids(text)

def bpe_encoding_ids_with_eos(func,text):
    return func.encode_ids_with_eos(text)

def bpe_decoding_ids(func,ids):
    return func.decode_ids(ids)


def text_ids_norm(ids,max_len):
    n = min(len(ids), max_len)
    ids = ids if len(ids) <= max_len else ids[:max_len]
    tmp = [EOS_ID] * max_len
    tmp[:n] = ids
    return tmp

def text_ids_norm2(ids,max_len):
    n = min(len(ids), max_len)
    ids = ids if len(ids) <= max_len else ids[:max_len]
    ids = ids[:-1]
    tmp = [EOS_ID] * max_len
    tmp[0] = SOS_ID
    tmp[1:n] = ids
    return tmp

def text_dct_ids(func,dct):
    for utt in dct:
        dct[utt] = bpe_encoding_ids(func,dct[utt])
    return dct

def text_dct_ids_with_eos(func,dct,max_len=None):
    for utt in dct:
        ans = bpe_encoding_ids_with_eos(func,dct[utt])
        if max_len:
            ans = text_ids_norm(ans,max_len)
            dct[utt] = ans
        else:
            dct[utt] = ans
    return dct


"""
============================
        GENERATOR 
============================
"""
def limit_trans_utts(lst,trans_dct,max_len=64):
    return [utt for utt in lst if len(trans_dct[utt_norm(utt)])<=max_len]

def limit_time_utts(lst, utt2frames, max_len=64):
    return [utt for utt in lst if int(utt2frames[utt]) <= max_len]

def load_ctc_accent(lst, feats,
                    max_label_len,
                    max_input_len,
                    encoder_len,
                    accent_classes,
                    trans_ids,
                    accent_dct,
                    accent_ids, ):
    inputs1 = []
    ctc_input_len = []
    ctc_label_len = []
    ctc_labels = []
    accent_labels = []
    for utt in lst:
        utt = utt_norm(utt)
        ctc_label = trans_ids[utt]
        ctc_label_norm = text_ids_norm(ctc_label, max_label_len)
        inputs1.append(feat_reshape(feat_norm(feats[utt]),max_input_len))
        ctc_input_len.append(encoder_len)
        ctc_label_len.append(min(len(ctc_label)-1,max_label_len) )
        ctc_labels.append(ctc_label_norm)
        accent_labels.append(to_categorical(accent_ids[accent_dct[utt]], num_classes=accent_classes))
    return {"inputs": np.float32(np.expand_dims(np.asarray(inputs1), axis=3)),
            "ctc_input_len": np.int32(np.expand_dims(np.asarray(ctc_input_len), axis=1)),
            "ctc_label_len": np.int32(np.expand_dims(np.asarray(ctc_label_len), axis=1)),
            "ctc_labels": np.float32(np.asarray(ctc_labels))}, \
           {"ctc_loss": np.zeros([len(lst)]),
            "accent_labels": np.asarray(accent_labels)}


# For CTC and Accent
def generator_ctc_accent(lst, feats, batch_size,
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
            data, labels = load_ctc_accent(subs, feats,
                                           max_label_len,
                                           max_input_len,
                                           encoder_len,
                                           accent_classes,
                                           trans_ids,
                                           accent_dct,
                                           accent_ids )
            yield data, labels


# For CTC
def load_ctc(lst, feats,
            max_label_len,
            max_input_len,
            encoder_len,
             trans_ids):
    inputs1 = []
    ctc_input_len = []
    ctc_label_len = []
    ctc_labels = []
    for utt in lst:
        utt = utt_norm(utt)
        ctc_label = trans_ids[utt]
        ctc_label_norm = text_ids_norm(ctc_label, max_label_len)
        inputs1.append(feat_reshape(feat_norm(feats[utt]),max_input_len))
        ctc_input_len.append(encoder_len)
        ctc_label_len.append(min(len(ctc_label)-1,max_label_len) )
        ctc_labels.append(ctc_label_norm)
    return {"inputs": np.float32(np.expand_dims(np.asarray(inputs1), axis=3)),
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



# For transformer
def load_tfr(lst, feats,
            max_label_len,
            max_input_len,
            bpe_classes,
            trans_ids):
    encoder_inputs = []
    decoder_inputs = []
    outputs = []
    for utt in lst:
        utt = utt_norm(utt)
        label = trans_ids[utt]
        label_norm = text_ids_norm(label, max_label_len)
        label_norm2 = text_ids_norm2(label, max_label_len)
        label_norm_oh = to_categorical(label_norm,bpe_classes)
        encoder_inputs.append(feat_reshape(feat_norm(feats[utt]),max_input_len))
        decoder_inputs.append(label_norm2)
        outputs.append(label_norm_oh)
    return {"encoder_input": np.float32(np.asarray(encoder_inputs)),
            "decoder_input": np.float32(np.asarray(decoder_inputs))}, \
           {"output": np.float32(np.asarray(outputs))}


def generator_tfr(lst, feats, batch_size,
                  max_label_len,
                  max_input_len,
                  bpe_classes,
                  trans_ids
                  ):
    n_batchs = len(lst) // batch_size
    while True:
        random.shuffle(lst)
        for i in range(n_batchs):
            begin = i * batch_size
            end = begin + batch_size
            subs = lst[begin:end]
            data, labels = load_tfr(subs, feats,
                                    max_label_len,
                                    max_input_len,
                                    bpe_classes,
                                    trans_ids)
            yield data, labels



# pred
def feats2inputs(feats,max_input_len):
    utts = []
    arr = []
    for utt in feats:
        utt = utt_norm(utt)
        utts.append(utt)
        arr.append(feat_reshape(feat_norm(feats[utt]), max_input_len))
    return utts, np.expand_dims(np.asarray(arr),axis=3)

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

def ctc_eval(labels, label_lens, preds, show=None):
    wer_,count = 0,0
    for label,length,pred in zip(labels,label_lens,preds):
        label = [int(item) for item in label]
        length = int(length[0])
        true = label[:length]
        pred = [int(item) for item in pred if item != -1]
        true_ = BPE_EN.decode_ids(true)
        pred_ = BPE_EN.decode_ids(pred) if pred else ""
        wer = edit_distance(true_.split(),pred_.split())/len(true_.split())
        wer_ += wer
        count += 1
    if show:
        print("DEMO-REF", true, true_)
        print("DEMO-HYP", pred, pred_)
    return wer_/count

def softmax2res(ref,hyp,show=True):
    refs = np.argmax(ref, axis=2)
    hyps = np.argmax(hyp, axis=2)
    wer_,count = 0,0
    for ref,hyp in zip(refs,hyps):
        ref_ = BPE_EN.decode_ids(ref).split()
        hyp_ = BPE_EN.decode_ids(hyp).split()
        wer = edit_distance(ref_,hyp_)/len(ref_)
        wer_ += wer
        count += 1
    if show:
        print("DEMO-REF", ref, ref_)
        print("DEMO-HYP", hyp, hyp_)
    return wer_/count

def acc(ref,hyp,accent=-1):
    ref = np.argmax(ref, axis=1)
    hyp = np.argmax(hyp, axis=1)
    if accent>-1:
        ids = np.where(ref==accent)[0]
        return sum(ref[ids] == hyp[ids])/len(ids)
    else:
        return sum(ref==hyp)/len(ref)

"""
================= 
    PREDEFINATION 
=================
"""
# print("------------ loading predefination ---------------------")
# AESRC_ACCENT = scp2dct(read_lines("src/AESRC2020/NATION.TXT"))
# AESRC_ACCENT2INT = {"Chinese":0, "Japanese":1, "Indian":2, "Korean":3, "American":4, "British":5, "Portuguese":6, "Russian":7}
# AESRC_TRANS = filt_text_dct(scp2dct(read_lines("/disc1/AESRC2020/data/aesrc_fbank/trans.scp")))
# AESRC_TRANS_IDS = text_dct_ids_with_eos(BPE_EN, AESRC_TRANS)
# AESRC_UTT2FRAMES = scp2dct(read_lines("/disc1/AESRC2020/data/aesrc_fbank_sp/utt2num_frames"))
# LIBRI_TRANS = filt_text_dct(scp2dct(read_lines("/disc1/AESRC2020/data/librispeech_train/trans.scp")))
# LIBRI_TRANS_IDS = text_dct_ids_with_eos(BPE_EN, LIBRI_TRANS)
# LIBRI_UTT2FRAMES = scp2dct(read_lines("/disc1/AESRC2020/data/librispeech_train/utt2num_frames"))
# save('data/dict/aesrc_nation.pkl', AESRC_ACCENT)
# save('data/dict/aesrc_nation_int.pkl', AESRC_ACCENT2INT)
# save('data/dict/aesrc_trans_ids.pkl',AESRC_TRANS_IDS)
# save('data/dict/aesrc_utt2frames.pkl',AESRC_UTT2FRAMES)
# save('data/dict/libri_trans_ids.pkl',LIBRI_TRANS_IDS)
# save('data/dict/libri_utt2frames.pkl',LIBRI_UTT2FRAMES)

AESRC_ACCENT = load('/disc1/AESRC2020/data/dict/aesrc_nation.pkl')
AESRC_ACCENT2INT = load('/disc1/AESRC2020/data/dict/aesrc_nation_int.pkl')
AESRC_TRANS_IDS = load('/disc1/AESRC2020/data/dict/aesrc_trans_ids.pkl')
AESRC_UTT2FRAMES = load('/disc1/AESRC2020/data/dict/aesrc_utt2frames.pkl')
LIBRI_TRANS_IDS = load('/disc1/AESRC2020/data/dict/libri_trans_ids.pkl')
LIBRI_UTT2FRAMES = load('/disc1/AESRC2020/data/dict/libri_utt2frames.pkl')
M,V = get_cmvn('/disc1/AESRC2020/data/aesrc_fbank_sp/cmvn.ark')

if __name__ == "__main__":

    aesrc_scp_file = "/disc1/AESRC2020/data/aesrc_fbank_sp/feats.scp"
    libri_scp_file = "/disc1/AESRC2020/data/librispeech_train/feats.scp"


    aesrc_lst = limit_time_utts(limit_trans_utts(scp2key(read_lines(aesrc_scp_file)), AESRC_TRANS_IDS, 100), AESRC_UTT2FRAMES, 1600)
    libri_lst = limit_time_utts(limit_trans_utts(scp2key(read_lines(libri_scp_file)), LIBRI_TRANS_IDS, 100), LIBRI_UTT2FRAMES, 1600)

    AESRC_FEATS = kaldiio.load_scp(aesrc_scp_file)
    LIBRI_FEATS = kaldiio.load_scp(libri_scp_file)

    generator = generator_ctc_accent(aesrc_lst, AESRC_FEATS, 32,
                                     max_input_len=1200,
                                     max_label_len=72,
                                     encoder_len=114,
                                     accent_classes=8,
                                     trans_ids=AESRC_TRANS_IDS,
                                     accent_dct=AESRC_ACCENT,
                                     accent_ids=AESRC_ACCENT2INT)

    data = next(generator)

    exit()

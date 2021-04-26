import pickle
from random import shuffle, randrange
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()

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


"""
==========================
        FEATURE OPT 
==========================
"""
def feat_norm(feat):
   return MMS.fit_transform(feat)


def feat_reshape(feat, max_len=1200):
    h, w = feat.shape
    if h >= max_len:
        return feat[:max_len]
    else:
        feat_ = np.zeros((max_len, w))
        feat_[:h] = feat
        return feat_

"""
==========================
     TRANSCRIPT OPT 
==========================
"""
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def text_ids_norm(ids,max_len):
    if len(ids)>max_len:
        ids = ids[:max_len]
    n = min(len(ids), max_len)
    tmp = [EOS_ID] * max_len
    tmp[:n] = ids
    return tmp


"""
==========================
        Generator 
==========================
"""
def data_loader(lst,
                ctc_enable=False,
                ar_enable=False,
                disc_enable=False,
                data_dct = None,
                accent_dct = None,
                trans_dct = None,
                max_input_len=1200,
                max_ctc_len=72,
                encoder_len=100,
                accent_classes=8,
                bn=0
                ):
    inputs = []
    ctc_input_len = []
    ctc_label_len = []
    ctc_labels = []
    accent_labels = []
    for utt in lst:
        # input features
        inputs.append(feat_reshape(feat_norm(load(data_dct[utt])), max_input_len))
        # ctc labels
        if ctc_enable and trans_dct:
            trans = trans_dct[utt]
            trans_norm = text_ids_norm(trans, max_ctc_len)
            ctc_input_len.append(encoder_len)
            ctc_label_len.append(min(len(trans),max_ctc_len))
            ctc_labels.append(trans_norm)
        if ar_enable and accent_dct:
            accent_labels.append(to_categorical(int(accent_dct[utt]), num_classes=accent_classes))

    input_data = {"x_data": np.float32(np.expand_dims(np.asarray(inputs), axis=3))}
    output_data = {}
    if ctc_enable:
        input_data["x_ctc_in_len"] = np.int32(np.expand_dims(np.asarray(ctc_input_len), axis=1))
        input_data["x_ctc_out_len"] = np.int32(np.expand_dims(np.asarray(ctc_label_len), axis=1))
        input_data["x_ctc_label"] = np.float32(np.asarray(ctc_labels))
        output_data["y_ctc_loss"] = np.zeros([len(lst)])
    if ar_enable:
        output_data["y_accent"] = np.asarray(accent_labels)
    if disc_enable:
        input_data["x_accent"] = np.asarray(accent_labels)
        output_data["y_disc"] = np.asarray(accent_labels)
        if bn:
            output_data["y_disc_bn"] = np.asarray(accent_labels)
    return input_data, output_data



def data_generator(lst,
                   ctc_enable = False,
                   ar_enable = False,
                   disc_enable = False,
                   batch_size = 32,
                   data_dct = None,
                   accent_dct = None,
                   trans_dct = None,
                   max_input_len=1200,
                   max_ctc_len=72,
                   encoder_len=100,
                   accent_classes=8,
                   bn=0,
                   ):
    n_batchs = len(lst) // batch_size
    while True:
        shuffle(lst)
        for i in range(n_batchs):
            begin = i * batch_size
            end = begin + batch_size
            sub = lst[begin:end]
            input,output = data_loader(sub,
                                        ctc_enable=ctc_enable,
                                        ar_enable=ar_enable,
                                        disc_enable=disc_enable,
                                        data_dct = data_dct,
                                        accent_dct = accent_dct,
                                        trans_dct = trans_dct,
                                        max_input_len=max_input_len,
                                        max_ctc_len=max_ctc_len,
                                        encoder_len=encoder_len,
                                        accent_classes=accent_classes,
                                        bn=bn
                                       )
            yield input,output

def cal_descriptors(T,D):
    def pool(x):
        return np.ceil(x/2)
    return int(pool(pool(pool(pool(pool(T)))))*pool(pool(pool(pool(pool(D))))))



if __name__ == "__main__":
    lst = read_lines("dev.lst")
    data_dct = load("array/data_scp.pkl")
    trans_dct = load("array/trans_scp.pkl")
    accent_dct = load("array/accent_scp.pkl")
    # input,output = data_loader(lst,
    #                             ctc_enable=True,
    #                             ar_enable=True,
    #                             disc_enable=True,
    #                             data_dct = data_dct,
    #                             accent_dct = accent_dct,
    #                             trans_dct = trans_dct,
    #                             max_input_len=1200,
    #                             max_ctc_len=72,
    #                             encoder_len=100,
    #                             accent_classes=8,
    #                            )
    generator = data_generator(lst,
                                batch_size=32,
                                ctc_enable=True,
                                ar_enable=True,
                                disc_enable=True,
                                data_dct = data_dct,
                                accent_dct = accent_dct,
                                trans_dct = trans_dct,
                                max_input_len=1200,
                                max_ctc_len=72,
                                encoder_len=100,
                                accent_classes=8,)
    data = next(generator)
    print(cal_descriptors(1200,80))
    import matplotlib.pyplot as plt
    plt.imshow(data[0]['x_data'][0])
    plt.waitforbuttonpress(0)
    exit()

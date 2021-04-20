from bpemb import BPEmb
import sys
import re 
import pickle

"""
================= 
      BPE 
=================
"""
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2
BPE_EN_1K = BPEmb(lang="en", vs=1000)

def bpe_encoding_ids_with_eos(func,text):
    return func.encode_ids_with_eos(text)
    

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


"""
=========================== 
        FILE 
===========================
"""
def save(path,arr):
    f = open(path,'wb')
    pickle.dump(arr,f)
    f.close()

if __name__ == "__main__":

    dct = {}
    while True:
        line = sys.stdin.readline().strip()
        if line == '':break
        utt = line.split()[0]
        value = ' '.join(line.split()[1:]).upper()
        value = filt_text(value)
        value = bpe_encoding_ids_with_eos(BPE_EN_1K,value)
        dct[utt] = value
    save(sys.argv[1],dct)
        
        
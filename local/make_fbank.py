import python_speech_features as psf
import soundfile as sf
# import scipy.io.wavfile as wav
import pickle as pkl
import sys
import os
import re

# linux to windows 路径转换
def path_lin2win(path):
    pattern = "/[a-z]/"
    position = re.findall(pattern,path)[0][1].upper()
    return re.sub(pattern,"%s:/"%position,path)

# 存储文件
def save(data,path):
    f = open(path,"wb")
    pkl.dump(data,f)
    f.close()

def path2utt(path):
    return path.split('/')[-1].split('.')[0]

def fbank(path):
    # path = path_lin2win(path) # windows path
    y,sr = sf.read(path)
    mel = psf.fbank(y,samplerate=sr,nfilt=80)[0]
    return mel

if __name__ == "__main__":
    audio_file = sys.argv[1]
    # audio_file = r"E:/LIBRISPEECH/LibriSpeech/dev/dev-clean/1272/128104/1272-128104-0000.flac"
    out_file = sys.argv[2]
    dir = os.path.dirname(out_file)
    if not os.path.isdir(dir):os.mkdir(out_file)
    
    mel = fbank(audio_file)
    save(mel,out_file)
    print(path2utt(out_file),mel.shape[0])

    exit()



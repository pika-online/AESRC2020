import sys
import pickle


def load(path):
    f = open(path,'rb')
    return pickle.load(f)

if __name__ == "__main__":
    frames_dct = load(sys.argv[1])
    trans_dct = load(sys.argv[2])
    max_input = int(sys.argv[3])
    max_output = int(sys.argv[4])

    while True:
        line = sys.stdin.readline().strip()
        if line == '':break
        utt = line.split()[0]
        if utt not in frames_dct:continue
        frame = int(frames_dct[utt])
        trans = trans_dct[utt]
        if frame<=max_input and len(trans)<max_output:
            # print(frame,trans)
            print(utt)


import sys
import pickle

def save(path,arr):
    f = open(path,'wb')
    pickle.dump(arr,f)
    f.close()

dct = {}
while True:
	line = sys.stdin.readline().strip()
	if line == '':break
	utt = line.split()[0]
	value = ' '.join(line.split()[1:])
	dct[utt] = value
save(sys.argv[1],dct)

set -e


##############################
#	(1) Data Prep (SCP)
##############################
## (i) audio list and scp
src=E:/AESRC2020/speech
test=E:/AESRC2020/test
dev_speaker=/e/AESRC2020/dev_speaker.lst
bash local/search.sh $src wav lst | sort | uniq > scp/src.lst
bash local/search.sh $test wav scp | sort | uniq > scp/test.scp
cat scp/src.lst | awk -F '/' '{a=$NF;sub(/\..*$/,"",a);printf("%s_%s %s\n",$5,a,$0)}' > scp/src.scp
cat scp/src.scp | grep -f $dev_speaker > scp/dev.scp
cat scp/src.scp | grep -v -f $dev_speaker > scp/train.scp

## (ii) accent labels
test_accent=E:/AESRC2020/utt2utt.txt
cat scp/src.lst | awk -F '/' 'BEGIN{ACCENT["Chinese"]="0";
ACCENT["Indian"]="1";
ACCENT["Japanese"]="2";
ACCENT["Korean"]="3";
ACCENT["American"]="4";
ACCENT["British"]="5";
ACCENT["Portuguese"]="6";
ACCENT["Russian"]="7";};{a=$NF;sub(/\..*$/,"",a);printf("%s_%s %s\n",$5,a,ACCENT[$5])}' > scp/src_accent.scp
cat $test_accent | awk 'BEGIN{ACCENT["CHN"]="0";
ACCENT["IND"]="1";
ACCENT["JPN"]="2";
ACCENT["KR"]="3";
ACCENT["US"]="4";
ACCENT["UK"]="5";
ACCENT["PT"]="6";
ACCENT["RU"]="7";
ACCENT["CAN"]="8";
ACCENT["ES"]="9";
};{a=$1;sub(/\-.*/,"",a);printf("%s %s\n",$2,ACCENT[a])}' > scp/test_accent.scp

## (iii) transcript labels
test_trans=E:/AESRC2020/2020AESRC评测测试集抄本.txt
nj=10
bash local/search.sh $src txt lst | sort | uniq > scp/src_txt.lst
bash local/multi_jobs.sh scp/src_txt.lst local/read_trans.sh "bash" $nj >> scp/src_trans.scp # Speed up reading with multijobs.
cat $test_trans > scp/test_trans.scp


##############################
#	(2) Scp2Npy (for python)
##############################
## (i) Fbank feature
nj=10
out_dir=array
cat scp/src.scp scp/test.scp | awk -v dir=$out_dir/speech '{printf("%s %s/%s.pkl\n",$2,dir,$1)}' > temp/fbank.params
bash local/multi_jobs.sh temp/fbank.params local/make_fbank.py "python" $nj > scp/utt2frames.scp
cat scp/src.scp scp/test.scp | awk -v dir=$out_dir/speech '{printf("%s %s/%s.pkl\n",$1,dir,$1)}' | python local/scp2dct.py $out_dir/data_scp.pkl
cat scp/utt2frames.scp | python local/scp2dct.py $out_dir/utt2frames_scp.pkl

## (ii) Accent Labels
cat scp/src_accent.scp scp/test_accent.scp | python local/scp2dct.py $out_dir/accent_scp.pkl

## (iii) transcription labels
cat scp/src_trans.scp scp/test_trans.scp | python local/bpe1k_encoding.py $out_dir/trans_scp.pkl

## (iv) limitation on train-set
max_input_len=1200
max_ctc_len=72
cat scp/train.scp | python local/limit.py $out_dir/utt2frames_scp.pkl $out_dir/trans_scp.pkl $max_input_len $max_ctc_len > train.lst #123975/123984
cat scp/dev.scp | python local/limit.py $out_dir/utt2frames_scp.pkl $out_dir/trans_scp.pkl $max_input_len $max_ctc_len > dev.lst # 12545/12545
cat scp/test.scp | python local/limit.py $out_dir/utt2frames_scp.pkl $out_dir/trans_scp.pkl $max_input_len $max_ctc_len > test.lst # 12545/12545


##############################
#	      (3) Training
##############################
python train.py


## AR-Net: Accent Recognition Network (Keras)
For Interspeech2020 Accented English Speech Recognition Challenges 

##### Author: Ephemeroptera
##### Blog: https://blog.csdn.net/Ephemeroptera/article/details/108680076
##### Date: 2020-09-25
##### Keywords: e2e, resnet, multi-task-learning

##### 1. Introduction
Accent recognition is closely related to speech recognition, It is easy to fall into the overfitting situation if we only do simple accent classification,
hence we introduce speech recognition task to build a multi-task model.

##### 2. Architecture

We adopt CNN + RNN encoding framework and ASR/AR multi-task outputs:
![avatar](https://img-blog.csdnimg.cn/20200930124456515.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)

The above model can be summarized as follow:
    
    <INPUTS>: [N,MAX_TIME,80,1] # 80-dim fbank features from kaldi-tools (+ CMN)
    <ENCODER> resnet(Generating internal features and Pooling) + RNN(seq2seq model)
    <OUTPUTS:CTC> for e2e-ASR (ctc loss)
    <OUTPUTS:ACCENT> for accent classification (CE)
    
##### 3. Speech Data
###### 3.1 Aesrc data: 160 hours Accented English Data 
The DataTang will provide participants with a total of 160 hours of English data collected from eight countries:
    
    Russian, 
    Korean, 
    American, 
    Portuguese, 
    Japanese, 
    Indian, 
    British 
    Chinese  
with about 20 hours of data for each accent
###### 3.2 Auxiliary Data: 1000hours Librispeech corpus
Librispeech data consists of 960 hours of training data and 40 hours of test data, you can obtain from: http://www.openslr.org/12/



##### 4. Training Method

###### 4.1 training/network config

EPOCHS| INIT_LR  | BPE_SIZE |
|----|----|----|
20|0.001|1000|


| MAX_SEQ_LEN (libri) | MAX_LABEL_LEN (libri) | ENCODER_LEN (libri) |
|----|----|----|
|1600 | 100 |150 |


|  MAX_SEQ_LEN (aesrc)| MAX_LABEL_LEN (aesrc)  | ENCODER_LEN (aesrc) |
|----|----|----|
|  1200| 72 |114|

Training_tricks: ReduceLROnPlateau, EarlyStopping (libri_monitor:dev_loss, aesrc_monitor: dev_acc)

###### 4.2 pre-training : Initialize the hidden layer (librispeech)
 The red outline represents the initialized weights by librispeech-ctc-training
![avatar](https://img-blog.csdnimg.cn/20200930124919696.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_cente)

###### 4.3 training ARNet


##### 5. Results
###### 5.1 librispeech
CTC WER (no lm):

|  |dev_clean | dev_other | test_clean | test_other |
|----|----|----|----|----|
| resnet18 + bi-gru| 20.7% |37.5%|20.9%|38.6%|

**Experiments have shown that using a deeper resnet can further reduce word error rates,
such as: resnet34 brought 16.7% wer in dev_clean**



###### 5.2 aesrc
CTC WER (no lm):

|  |dev  | test| 
|----|----|----|
|  resnet18 + bi-gru| 24% |-|-|-|


 
Accent Acc (dev):
 
| |  Chinese|Japanese  |India| Korea | American | Britain | Portuguese| Russia| Overall
|----|----|----|----|----|----|----|----|----|----|
| resnet18 + bi-gru|  0.64| 0.69 |0.97|0.66|0.58|0.92|0.82|0.70|**0.75**



###### 5.3 Official Baseline
Officials have also provided a good baseline: https://github.com/R1ckShi/AESRC2020, That method is based on ESPNET, and the model consists of Transformer and ASR-init.

Accent acc (dev):

|  Chinese|Japanese  |India| Korea | American | Britain | Portuguese| Russia| Overall
|----|----|----|----|----|----|----|----|----|
|  0.67| 0.73 |0.97|0.56|0.60|0.94|0.86|0.76|0.76




Welcome to fork and star ~

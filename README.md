
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

We tried the following models
###### 2.1 ARNet-A : CNN + RNN framework 
![avatar](https://img-blog.csdnimg.cn/20200928194607729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)
###### 2.2 ARNet-B : CNN + Attention framework
![avatar](https://img-blog.csdnimg.cn/20200928213059726.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)
###### 2.3 ARNet-C : CNN + Transformer framework
![avatar](https://img-blog.csdnimg.cn/20200928213123878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)    

The above model can be summarized as follow:
    
    <INPUTS>: [N,MAX_TIME,80,1] # 80-dim fbank features from kaldi-tools (+ CMN)
    <ENCODER> resnet(Generating internal features and Pooling) + RNN/Attention/Transformer(seq2seq task)
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

|  GPUS|BATCH_SIZE  |EPOCHS| INIT_LR |BPE_SIZE | CNN |
|----|----|----|----|----|----|
|  3| 300 |30|0.001|1000|resnet-18|


| MAX_SEQ_LEN (libri) | MAX_LABEL_LEN (libri) | ENCODER_LEN (libri) |
|----|----|----|
|1600 | 100 |150 |


|  MAX_SEQ_LEN (aesrc)| MAX_LABEL_LEN (aesrc)  | ENCODER_LEN (aesrc) |
|----|----|----|
|  1200| 72 |114|

Training_tricks: ReduceLROnPlateau, EarlyStopping

###### 4.2 pre-training : Initialize the hidden layer (librispeech)
Take model ARNet-A as example: the red outline represents the initialized weights by librispeech-ctc-training
![avatar](https://img-blog.csdnimg.cn/20200928214930376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)

###### 4.3 training ARNet
Multi-task Loss weight setting:

|   | Loss(ctc) : Loss(accent)|
|----|----|
|  ARNet-A| 0.7 : 0.3 |
|  ARNet-B| 0.5 : 0.5|
|  ARNet-C| 0.5 : 0.5|


##### 5. Results
###### 5.1 Pre-training
    Corpus: Librispeech
    INPUT_SIZE: 1600x80
    ENCODER_LEN: 150
    MAX_CTC_LABEL_LEN: 100
    FRONT-CNN: resnet-18
    BPE: 1000 subwords
    EPOCHS: 10
    CTC-WER(dev_clean): 25%

###### 5.2 AR-Net
    Corpus: AESRC
    INPUT_SIZE: 1200x80
    ENCODER_LEN: 114
    MAX_CTC_LABEL_LEN: 72
    FRONT-CNN: resnet-18
    LOSS-WEIGHTS: 0.7(CTC) + 0.3(Accent) --> 0.4(CTC) + 0.6(Accent) [finetune]
    BPE: 1000 subwords
    EPOCHS: 10
    CTC-WER(dev): 29%
    Accent-ACC(dev): 72%
 
 Accent Acc:
 
|  Chinese|Japanese  |India| Korea | American | Britain | Portuguese| Russia| Overall
|----|----|----|----|----|----|----|----|----|
|  0.56| 0.70 |0.96|0.67|0.49|0.88|0.79|0.71|0.72

![avatar](https://img-blog.csdnimg.cn/20200926203352622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)


###### 5.3 Official Baseline
Officials have also provided a good baseline: https://github.com/R1ckShi/AESRC2020, That method is based on ESPNET, and the model consists of Transformer and ASR-init.

|  Chinese|Japanese  |India| Korea | American | Britain | Portuguese| Russia| Overall
|----|----|----|----|----|----|----|----|----|
|  0.67| 0.73 |0.97|0.56|0.60|0.94|0.86|0.76|0.76




Welcome to fork and star ~

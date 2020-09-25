
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

![avatar](https://img-blog.csdnimg.cn/20200925222612861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)

    <INPUTS> [N,MAX_TIME,80,1] # 80-dim fbank features from kaldi-tools
    <SHARED-HIDDEN> resnet-18/34/50/101/152 + Bi-GRU
    <OUTPUTS:CTC> for e2e-ASR (ctc loss)
    <OUTPUTS:ACCENT> for accent recognition (CE)
    
##### 3. Speech Data
###### 3.1 Accented English Data
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
###### 3.2 Auxiliary Data: Librispeech (1000 hours)
Librispeech data consists of 960 hours of training data and 40 hours of test data, you can obtain from: http://www.openslr.org/12/

##### 4. Training Method
###### 4.1 Pre-training : Initialize the hidden layer (librispeech)
![avatar](https://img-blog.csdnimg.cn/20200925230950457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)

###### 4.2 Training CTC-Accent Model
![avatar](https://img-blog.csdnimg.cn/20200925231414601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VwaGVtZXJvcHRlcmE=,size_16,color_FFFFFF,t_70#pic_center)
the red line represents the initialized hidden layer 

###### 5. Results
from local.resnet import resnet18_,resnet34_,resnet50_,resnet101_,resnet152_
from keras.layers import Input, Dense, Lambda,Dropout, Reshape,LSTM,Conv2D,MaxPooling2D,Activation,\
    Bidirectional,GRU,GlobalAveragePooling1D,Layer,AveragePooling1D,BatchNormalization,Add,MaxPooling1D,Conv1D,\
    AveragePooling2D,GlobalAveragePooling2D
from keras_layer_normalization import LayerNormalization
from keras.layers.cudnn_recurrent import CuDNNGRU,CuDNNLSTM
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras.utils import multi_gpu_model
from keras.optimizers import Adam,Nadam,SGD
import local.losses as ls
import local.VLAD as vd
from local.attention import Attention

"""
=========================
        layers
=========================
"""

def SQUEEZE(axis=3, name=None):
    return Lambda(lambda x: K.squeeze(x,axis=axis),name=name)

def EXPAND(axis=3,name=None):
    return Lambda(lambda x: K.expand_dims(x, axis=axis),name=name)

def BN(name=None):
    return BatchNormalization(name=name)



def LN(name=None):
    return LayerNormalization(name=name)

def DS(hidden,activation,rgr=l2(1e-4),use_bias=True,name=None):
    return Dense(hidden,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer='he_normal',
                 kernel_regularizer=rgr,
                 bias_regularizer=rgr,
                 name=name)

def BIGRU(hidden,seq=True,rgr=l2(1e-4),name=None):
    return Bidirectional(CuDNNGRU(hidden,
                                  return_sequences=seq,
                                  kernel_regularizer=rgr,
                                  bias_regularizer=rgr),
                         merge_mode='concat', name=name)

def cnn2seq(cnn_tensor):
    _, h, w, c = K.int_shape(cnn_tensor)
    encoder_len = int(h * w)
    print("==== the encoder length is: %d ====" % encoder_len)
    print("==== Warnning: you should let max_label_len <= encoder_len ====")
    return K.reshape(cnn_tensor,[-1,encoder_len,c])

def CNN2SEQ(name=None):
    return Lambda(function=cnn2seq,name=name)

def DP(rate,name=None):
    return Dropout(rate,name=name)


def BN_RELU(x):
    x = BN()(x)
    return Activation('relu')(x)

def LN_RELU(x):
    x = LN()(x)
    return Activation('relu')(x)


"""
=========================
        ctc constructors
=========================
"""

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_module(ctc_pred,max_label_len):
    ctc_input_len = Input(shape=[1], dtype='int32', name='ctc_input_len')
    ctc_label_len = Input(shape=[1], dtype='int32', name='ctc_label_len')
    ctc_labels = Input([max_label_len], dtype='float32', name='ctc_labels')
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')\
                ([ctc_pred, ctc_labels, ctc_input_len, ctc_label_len])

    return ctc_loss,ctc_labels, ctc_input_len, ctc_label_len



"""
=========================
        VLAD
=========================
"""
def vlad(x,
         aggregation,
         vlad_clusters,
         ghost_clusters):
    weight_decay = 1e-4

    if aggregation == 'vlad':
        x_k_center = Conv2D(vlad_clusters, (1, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='vlad_center_assignment')(x)
        x = vd.VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = Conv2D(vlad_clusters + ghost_clusters, (1, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='gvlad_center_assignment')(x)
        x = vd.VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')(
            [x, x_k_center])

    return x



"""
=========================
        AR moudle
=========================
"""
def integration(x,
                hidden_dim=256,
                mto='avg',
                vlad_clusters=8,
                ghost_clusters=2):

    if mto== 'avg':
        x = GlobalAveragePooling1D(name="AR_MERGE")(x)

    elif mto== 'bigru':
        x = BIGRU(hidden_dim, seq=False, name="AR_MERGE")(x)

    elif mto == 'attention':
        x = Attention(hidden_dim,name="AR_MERGE")(x)

    elif mto in ['vlad', 'gvlad']:
        x = EXPAND(axis=1)(x)
        x = vlad(x,
                 aggregation=mto,
                 vlad_clusters=vlad_clusters,
                 ghost_clusters=ghost_clusters)
    else:
        pass
    return x


def metric_space(x,
                 accent_input,
                 acc_classes,
                 metric_loss,
                 margin,
                 name):

    if metric_loss == "softmax":
        y = DS(acc_classes,activation='softmax',name=name)(x)

    elif metric_loss == "sphereface":
        y = ls.SphereFace(n_classes=acc_classes, m=margin, name=name)([x, accent_input])

    elif metric_loss == "cosface":
        y = ls.CosFace(n_classes=acc_classes, m=margin, name=name)([x, accent_input])

    elif metric_loss == "arcface":
        y = ls.ArcFace(n_classes=acc_classes, m=margin, name=name)([x, accent_input])

    elif metric_loss == "circleloss":
        y = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = Dense(acc_classes,activation=None,use_bias=False,kernel_constraint=unit_norm(),name=name)(y)
    else:
        return

    return y


"""
=========================
        Models
=========================
"""


def build( inputs, outputs, raw=None, name="model"):
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()
    if raw:
        print("===== init weights from:%s =====" % raw)
        model.load_weights(raw, by_name=True, skip_mismatch=True)
    return model



def SAR_Net(input_shape,
            asr_enable = True,
            ar_enable = True,
            res_type="res18",
            res_filters=64,
            hidden_dim=256,
            bn_dim=0,
            encoder_rnn_num=1,
            asr_rnn_num=1,
            bpe_classes=1000,
            acc_classes=8,
            max_label_len=72,
            mto=None,
            vlad_clusters=8,
            ghost_clusters=2,
            metric_loss='cosface',
            margin=0.3,
            raw_model=None,
            name=None):

    # =========================
    #         INPUT
    # =========================
    inputs = Input(input_shape,name="inputs")
    accent_input = Input([acc_classes],name="accent_inputs")

    # =========================
    #      SHARED ENCODER
    # =========================
    if res_type == "res18":
        cnn = resnet18_(inputs, filters=res_filters)
    elif res_type == "res34":
        cnn = resnet34_(inputs, filters=res_filters)
    elif res_type == "res50":
        cnn = resnet50_(inputs, filters=res_filters)
    elif res_type == "res101":
        cnn = resnet101_(inputs, filters=res_filters)
    elif res_type == "res152":
        cnn = resnet152_(inputs, filters=res_filters)
    else:
        print("======= ERROR: please specify cnn in res-[18,34,50,101,152] ======")
    cnn = CNN2SEQ(name="RES2SEQ")(cnn)
    cnn = DS(hidden_dim, activation='tanh', name="RESNET_DS")(cnn)
    cnn = LN(name="RESNET_DS_LN")(cnn)
    x = cnn
    for i in range(encoder_rnn_num):
        x = BIGRU(hidden_dim, name="ENCODER_BIGRU_%d" % (i + 1))(x)
        x = LN(name="ENCODER_BIGRU_LN_%d"% (i + 1))(x)


    if asr_enable:
        # =========================
        #         ASR Branch
        # =========================
        asr = x
        for i in range(asr_rnn_num):
            asr = BIGRU(hidden_dim, name="CTC_BIGRU_%d" % (i + 1))(asr)
            asr = LN(name="CTC_BIGRU_LN_%d" % (i + 1))(asr)
        asr = DS(hidden_dim, activation='tanh', name='CTC_DS')(asr)
        asr = LN(name='CTC_DS_LN')(asr)
        # asr = DP(0.5,name="CTC_DP")(asr)
        ctc_pred = DS(bpe_classes, activation="softmax", name='ctc_pred')(asr)
        ctc_loss, ctc_labels, ctc_input_len, ctc_label_len = ctc_module(ctc_pred, max_label_len)



    if ar_enable:
        # =========================
        #   AR Branch: Integration
        # =========================
        ar = DS(hidden_dim,activation='tanh',name='AR_DS1')(x)
        ar = LN(name='AR_DS1_LN')(ar)
        ar = integration(ar,
                         hidden_dim=hidden_dim,
                         mto=mto,
                         vlad_clusters=vlad_clusters,
                         ghost_clusters=ghost_clusters)
        ar = BN(name='AR_BN1')(ar)
        # ar = DP(0.5,name="AR_DP")(ar)
        ar = DS(hidden_dim, activation=None, name="AR_EMBEDDING")(ar)
        ar = BN(name='AR_BN2')(ar)

        # ===================================
        #    AR Branch: Discriminative loss
        # ===================================
        ar1 = metric_space(ar,
                       accent_input=accent_input,
                       acc_classes=acc_classes,
                       metric_loss=metric_loss,
                       margin=margin,
                       name="accent_metric")

        # ========================================
        #    AR Branch: Visual BottleNeck feature
        # ========================================
        if bn_dim:
            bn = DS(64, activation='relu',name="AR_BN_DS1")(ar)
            bn = BN(name='AR_BN3')(bn)
            bn = DS(bn_dim, activation=None, name="bottleneck")(bn)
            bn = BN(name='AR_BN4')(bn)
            bn = metric_space(bn,
                            accent_input=accent_input,
                            acc_classes=acc_classes,
                            metric_loss=metric_loss,
                            margin=margin,
                            name="accent_bn")

        # =======================================
        #      AR Branch: Classification
        # =======================================
        ar2 = DS(64, activation='relu',name="AR_DS2")(ar)
        ar2 = DS(64, activation='relu',name="AR_DS3")(ar2)
        ar2 = DS(acc_classes,activation='softmax',name='accent_labels')(ar2)

    # ==============================
    #           Model
    # ==============================
    input_set = [inputs]
    output_set = []
    if ar_enable:
        input_set += [accent_input]
        output_set += [ar1,ar2]
    if asr_enable:
        input_set += [ctc_labels, ctc_input_len, ctc_label_len]
        output_set += [ctc_loss]
    if bn_dim:
        output_set += [bn]

    return build(inputs=input_set,outputs=output_set,raw=raw_model,name=name)






"""
=========================
       combile models
=========================
"""
def compile(model,
            gpus,
            lr,
            loss,
            loss_weights,
            metrics):
    if gpus>1:
        model_ = multi_gpu_model(model, gpus=gpus)
    else:
        model_ = model
    model_.compile(optimizer=Adam(lr,decay=2e-4),
                           loss=loss,
                           loss_weights=loss_weights,
                           metrics=metrics)
    return model_


"""
======================
        OTHER
======================

"""

def sub_model(model,input_name,output_name):
    inputs = model.get_layer(name=input_name).input
    outputs = model.get_layer(name=output_name).output
    return Model(inputs=inputs, outputs=outputs)

def ctc_pred(model,x,batch_size,input_len,):
    pred = model.predict(x,batch_size=batch_size)
    input_len = K.constant([input_len]*len(pred),dtype="int32")
    decoded = K.ctc_decode(pred, input_len, greedy=True, beam_width=100, top_paths=1)
    return K.get_value(decoded[0][0])

if __name__=="__main__":


    model = SAR_Net(input_shape=[1200, 80, 1],
                    ar_enable=True,
                    asr_enable=True,
                    res_type="res34",
                    res_filters=32,
                    hidden_dim=256,
                    encoder_rnn_num=1,
                    asr_rnn_num=1,
                    bpe_classes=1000,
                    acc_classes=8,
                    max_label_len=72,
                    mto='bigru',
                    vlad_clusters=8,
                    ghost_clusters=2,
                    metric_loss='arcface',
                    raw_model=None,
                    name=None)

    sub_model(model,'inputs','accent_labels')
    model.save_weights('../exp/demo.h5')
    model.load_weights('../exp/demo.h5')

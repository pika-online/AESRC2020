from resnet import resnet18_,resnet34_,resnet50_,resnet101_, resnet152_
from keras.layers import Input, Dense, Lambda,Dropout,Conv2D,Activation,Bidirectional,GlobalAveragePooling1D,\
    BatchNormalization,Reshape
from keras_layer_normalization import LayerNormalization
from keras.layers.cudnn_recurrent import CuDNNGRU,CuDNNLSTM
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import losses as ls
import VLAD as vd



"""
=========================
        Layers
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
                                  merge_mode='concat',
                                  name=name)

def DP(rate,name=None):
    return Dropout(rate,name=name)


"""
=========================
        ctc constructors
=========================
"""

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_module(ctc_pred,max_label_len):
    ctc_input_len = Input(shape=[1], dtype='int32', name='x_ctc_in_len')
    ctc_label_len = Input(shape=[1], dtype='int32', name='x_ctc_out_len')
    ctc_labels = Input([max_label_len], dtype='float32', name='x_ctc_label')
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='y_ctc_loss')\
                ([ctc_pred, ctc_labels, ctc_input_len, ctc_label_len])

    return ctc_loss,ctc_labels, ctc_input_len, ctc_label_len



"""
=========================
        NetVLAD
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
        AR Module
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

    elif mto in ['vlad', 'gvlad']:
        x = EXPAND(axis=1)(x)
        x = vlad(x,
                 aggregation=mto,
                 vlad_clusters=vlad_clusters,
                 ghost_clusters=ghost_clusters)
    else:
        print("Please specify avg/bigru/vlad/gvlad ..")
        exit(1)
    return x


def disc_loss(x,
              accent_label,
              accent_classes,
              loss,
              margin,
              name):

    if loss == "softmax":
        y = DS(accent_classes, activation='softmax', use_bias=False, name=name)(x)

    elif loss == "sphereface":
        y = ls.SphereFace(n_classes=accent_classes, m=margin, name=name)([x, accent_label])

    elif loss == "cosface":
        y = ls.CosFace(n_classes=accent_classes, m=margin, name=name)([x, accent_label])

    elif loss == "arcface":
        y = ls.ArcFace(n_classes=accent_classes, m=margin, name=name)([x, accent_label])

    elif loss == "circleloss":
        y = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = Dense(accent_classes, activation=None, use_bias=False, kernel_constraint=unit_norm(), name=name)(y)
    else:
        return

    return y


"""
=========================
        Model
=========================
"""
def build( inputs,
           outputs,
           raw=None,
           name="model"):
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()
    if raw:
        print("===== init weights from:%s =====" % raw)
        model.load_weights(raw, by_name=True, skip_mismatch=True)
    return model


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


def SAR_Net(input_shape,
            ctc_enable = False,
            ar_enable = True,
            disc_enable = False,
            res_type="res18",
            res_filters=64,
            hidden_dim=256,
            bn_dim=0,
            bpe_classes=1000,
            accent_classes=8,
            max_ctc_len=72,
            mto=None,
            vlad_clusters=8,
            ghost_clusters=2,
            metric_loss='cosface',
            margin=0.3,
            raw_model=None,
            lr=0.01,
            gpus = 1,
            mode="train",
            name=None):

    # =========================
    #   INPUT (2D Spectrogram)
    # =========================
    if mode=="train":
        inputs = Input(shape=input_shape,name="x_data")
    else:
        inputs = Input(shape=[None,input_shape[1],input_shape[2]], name="x_data")

    if disc_enable:
        disc_labels = Input(shape=(accent_classes,), name="x_accent")

    # ==============================
    #   SHARED ENCODER (Res + BiGRU)
    # ==============================
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
    cnn = Reshape([-1,K.int_shape(cnn)[-1]],name="CNN2SEQ")(cnn)
    cnn = DS(hidden_dim, activation='tanh', name="CNN_LIN")(cnn)
    cnn = LN(name="CNN_LIN_LN")(cnn)
    crnn = BIGRU(hidden_dim, name="CRNN")(cnn)
    crnn = LN(name="CRNN_LN")(crnn)

    # =========================
    #         ASR Branch
    # =========================
    if ctc_enable:
        asr = crnn
        asr = BIGRU(hidden_dim, name="CTC_BIGRU")(asr)
        asr = LN(name="CTC_BIGRU_LN")(asr)
        asr = DS(hidden_dim, activation='tanh', name='CTC_DS')(asr)
        asr = LN(name='CTC_DS_LN')(asr)

        ctc_pred = DS(bpe_classes, activation="softmax", name='ctc_pred')(asr)
        ctc_loss, ctc_labels, ctc_input_len, ctc_label_len = ctc_module(ctc_pred, max_ctc_len)


    # =========================
    #        AR Branch
    # =========================
    if ar_enable:
        # =========================
        #   AR Branch: Integration
        # =========================
        ar = DS(hidden_dim,activation='tanh',name='AR_DS')(crnn)
        ar = LN(name='AR_DS_LN')(ar)
        ar = integration(ar,
                         hidden_dim=hidden_dim,
                         mto=mto,
                         vlad_clusters=vlad_clusters,
                         ghost_clusters=ghost_clusters)
        ar = BN(name='AR_BN1')(ar)
        # ar = DP(0.5,name="AR_DP")(ar)
        ar = DS(hidden_dim, activation=None, name="AR_EMBEDDING")(ar) # Global Feature
        ar = BN(name='AR_BN2')(ar)

        # =======================================
        #      AR Branch: Classification
        # =======================================
        ar1 = DS(64, activation='relu',name="AR_CF_DS1")(ar)
        ar1 = DS(64, activation='relu',name="AR_CF_DS2")(ar1)
        ar1 = DS(accent_classes, activation='softmax', name='y_accent')(ar1)

        # ===================================
        #    AR Branch: Discriminative loss
        # ===================================
        if disc_enable:
            ar2 = disc_loss(ar,
                            accent_label=disc_labels,
                            accent_classes=accent_classes,
                            loss=metric_loss,
                            margin=margin,
                            name="y_disc")

        # ==========================================
        #    AR Branch: Visual BottleNeck feature (*)
        # ==========================================
        if disc_enable and bn_dim:
            bn = DS(64, activation='relu',name="AR_BN_DS")(ar)
            bn = BN(name='AR_BN3')(bn)
            bn = DS(bn_dim, activation=None, name="bottleneck")(bn)
            bn = BN(name='AR_BN4')(bn)
            bn = disc_loss(bn,
                           accent_label=disc_labels,
                           accent_classes=accent_classes,
                           loss=metric_loss,
                           margin=margin,
                           name="y_disc_bn")

    # ==============================
    #           Model
    # ==============================
    input_set = [inputs]
    output_set = []
    if ar_enable:
        output_set += [ar1]
    if disc_enable:
        input_set += [disc_labels]
        output_set += [ar2]
    if ctc_enable:
        input_set += [ctc_labels, ctc_input_len, ctc_label_len]
        output_set += [ctc_loss]
    if bn_dim:
        output_set += [bn]
    model = build(inputs=input_set,outputs=output_set,raw=raw_model,name=name)

    # ==============================
    #           Compile
    # ==============================
    loss = {}
    loss_weights = {}
    metrics = {}
    alpha = 0.4
    beta = 0.01
    if ar_enable:
        loss["y_accent"] = 'categorical_crossentropy'
        loss_weights["y_accent"] = beta if disc_enable else 1.0
        metrics["y_accent"] = "accuracy"
        if disc_enable:
            loss["y_disc"] = 'categorical_crossentropy' if metric_loss != 'circleloss' \
            else lambda y, x: ls.circle_loss(y, x, gamma=256, margin=margin)
            loss_weights["y_disc"] = 1-alpha if ctc_enable else 1.0
            metrics["y_disc"] = "accuracy"
    if ctc_enable:
        loss["y_ctc_loss"] = lambda y_true, y_pred: y_pred
        loss_weights["y_ctc_loss"] = alpha if disc_enable else 1.0
        loss_weights["y_ctc_loss"] = 1-alpha if not disc_enable else beta

    if bn_dim:
        loss["y_disc_bn"] = 'categorical_crossentropy' if metrics != 'circleloss' \
            else lambda y, x: ls.circle_loss(y, x, gamma=256, margin=margin)
        loss_weights["y_disc_bn"] = 0.1
        metrics['y_disc_bn'] = 'accuracy'

    train_model = compile(model,gpus,lr=lr,loss=loss,loss_weights=loss_weights,metrics=metrics)
    print(loss_weights)
    return model,train_model


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


    model,train_model = SAR_Net(input_shape=(1200,80,1),
                                ctc_enable = True,
                                ar_enable = True,
                                disc_enable = True,
                                res_type="res18",
                                res_filters=32,
                                hidden_dim=256,
                                bn_dim=0,
                                bpe_classes=1000,
                                accent_classes=8,
                                max_ctc_len=72,
                                mto='vlad',
                                vlad_clusters=8,
                                ghost_clusters=2,
                                metric_loss='cosface',
                                margin=0.3,
                                raw_model=None,
                                lr=0.01,
                                gpus = 1,
                                name=None)

    sub_model(model,'x_data','y_accent')
    model.save_weights('exp/demo.h5')
    model.load_weights('exp/demo.h5')

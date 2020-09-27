from local.resnet import resnet18_,resnet34_,resnet50_,resnet101_,resnet152_
from keras.layers import Input, Dense, Lambda,Dropout, Reshape,LSTM,Bidirectional,GRU
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.optimizers import Adam


"""
=========================
        ctc constructors
=========================
"""

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_input_length(tensor):
    return K.ones([K.int_shape(tensor)[0]],dtype='int32')*K.int_shape(tensor)[1]


"""
=========================
        models
=========================
"""

def model_ctc(shapes,
             bpe_classes,
             max_label_len,
             raw_model=None
             ):
    """
        Params:
            <shape>: the shape of input tensor
            <bpe_classes>: the size of bpe set
            <max_label_len>: maximum labels length (for ctc)
            <encoder_len>: encoder_length (for ctc)
            <raw_model>: init weights
        Return:
            <model>: ctc models
        Model:
            inputs: [inputs, ctc_labels, ctc_input_len,ctc_label_len]
            outputs: [ctc_loss]

        """

    # shared-layer
    cnn1 = resnet18_(shapes)
    # cnn1 = resnet34_(shapes)
    # cnn1 = resnet50_(shapes)
    # cnn1 = resnet101_(shapes)
    # cnn1 = resnet152_(shapes)
    encoder_len = int(cnn1.output.shape[1] * cnn1.output.shape[2])
    print("==== the model encoder length is: %d ====" % encoder_len)
    print("==== Warnning: you should let max_label_len <= encoder_len ====")
    cnn2 = Reshape((encoder_len, int(cnn1.output.shape[3])))
    cnn3 = Dense(256, activation="relu", kernel_initializer='he_normal', name="cnn")
    bigru1 = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(0.0001)),
                           merge_mode='sum', name="BIGRU1")
    bigru2 = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(0.0001)),
                           merge_mode='concat', name="BIGRU2")

    # INPUTS
    inputs = Input(shapes, name="inputs")

    # ENCODER
    encoder = cnn1(inputs)
    encoder = cnn2(encoder)
    encoder = cnn3(encoder)
    encoder = bigru1(encoder)
    encoder = Dropout(0.3)(encoder)

    # CTC
    ctc = bigru2(encoder)
    ctc = Dense(256, activation="relu", kernel_initializer='he_normal', name='ctc')(ctc)
    ctc = Dropout(0.3)(ctc)
    ctc_pred = Dense(bpe_classes, activation="softmax", kernel_initializer='he_normal', name='ctc_pred')(ctc)
    ctc_input_len = Input(shape=[1], dtype='int32', name='ctc_input_len')
    ctc_label_len = Input(shape=[1], dtype='int32', name='ctc_label_len')
    ctc_labels = Input([max_label_len], dtype='float32', name='ctc_labels')
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')(
        [ctc_pred, ctc_labels, ctc_input_len, ctc_label_len])


    # MODEL
    model = Model(inputs=[inputs, ctc_labels, ctc_input_len, ctc_label_len],
                  outputs=[ctc_loss], name="model_ctc")
    model.summary()
    if raw_model:
        print("===== init weights from:%s =====" % raw_model)
        model.load_weights(raw_model, by_name=True, skip_mismatch=True)

    return model


def model_ctc_accent(shapes,
                     bpe_classes,
                     accent_classes,
                     max_label_len,
                     raw_model=None
                     ):
    """
    Params:
        <shape>: the shape of input tensor
        <bpe_classes>: the size of bpe set
        <accent_classes>: the size of accents
        <max_label_len>: maximum labels length (for ctc)
        <encoder_len>: encoder_length (for ctc)
        <raw_model>: init weights
    Return:
        <model>: ctc-accents multi-task-learning models
    Model:
        inputs: [inputs, ctc_labels, ctc_input_len,ctc_label_len]
        outputs: [ctc_loss, accent_labels]

    """

    # shared-layer
    cnn1 = resnet18_(shapes)
    # cnn1 = resnet34_(shapes)
    # cnn1 = resnet50_(shapes)
    # cnn1 = resnet101_(shapes)
    # cnn1 = resnet152_(shapes)
    encoder_len = int(cnn1.output.shape[1]*cnn1.output.shape[2])
    print("==== the model encoder length is: %d ===="%encoder_len)
    print("==== Warnning: you should let max_label_len <= encoder_len ====")
    cnn2 = Reshape((encoder_len,int(cnn1.output.shape[3])))
    cnn3 = Dense(256, activation="relu", kernel_initializer='he_normal', name="cnn")
    bigru1 = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(0.0001)),
                             merge_mode='sum', name="BIGRU1")
    bigru2 = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(0.0001)),
                             merge_mode='concat', name="BIGRU2")

    # INPUTS
    inputs = Input(shapes, name="inputs")

    # ENCODER
    encoder = cnn1(inputs)
    encoder = cnn2(encoder)
    encoder = cnn3(encoder)
    encoder = bigru1(encoder)
    encoder = Dropout(0.3)(encoder)

    # CTC
    ctc = bigru2(encoder)
    ctc = Dense(256, activation="relu", kernel_initializer='he_normal', name='ctc')(ctc)
    ctc = Dropout(0.3)(ctc)
    ctc_pred = Dense(bpe_classes, activation="softmax", kernel_initializer='he_normal',name='ctc_pred')(ctc)
    ctc_input_len = Input(shape=[1], dtype='int32', name='ctc_input_len')
    ctc_label_len = Input(shape=[1], dtype='int32', name='ctc_label_len')
    ctc_labels = Input([max_label_len], dtype='float32', name='ctc_labels')
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([ctc_pred, ctc_labels, ctc_input_len, ctc_label_len])

    # ACCENT
    accents = Bidirectional(GRU(256, kernel_regularizer=l2(0.0001)), name="ACGRU")(encoder)
    accents = Dense(256, activation="relu", kernel_initializer='he_normal')(accents)
    accents = Dropout(0.3)(accents)
    aesrc_nation = Dense(accent_classes, activation='softmax', kernel_initializer='he_normal',name="accent_labels")(accents)

    # MODEL
    model = Model(inputs=[inputs, ctc_labels, ctc_input_len, ctc_label_len],
                  outputs=[ctc_loss, aesrc_nation ], name="model_ctc_accents")
    model.summary()
    if raw_model:
        print("===== init weights from:%s =====" % raw_model)
        model.load_weights(raw_model, by_name=True,skip_mismatch=True)

    return model



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
    parallel_model = multi_gpu_model(model, gpus=gpus)
    opt = Adam(lr)
    parallel_model.compile(optimizer=opt,
                           loss=loss,
                           loss_weights=loss_weights,
                           metrics=metrics)
    return parallel_model


"""
======================
    predict
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

    model = model_ctc(shapes=[1200,80,1],
                             bpe_classes=1000,
                             max_label_len=72)

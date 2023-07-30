import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model

def get_model_001():
    input_board = Input(shape=(8, 8, 12))
    input_meta = Input(shape=(5,))

    # layer arch
    board_conv1_layer = Conv2D(8, (3, 3), padding='same')
    board_conv1_bn_layer = BatchNormalization()
    board_conv1_activation_layer = Activation('relu')

    board_conv2_layer = Conv2D(16, (3, 3), padding='same', activation='relu')
    board_conv2_bn_layer = BatchNormalization()
    board_conv2_activation_layer = Activation('relu')
    board_conv2_attention_layer = Attention()

    board_conv3_layer = Conv2D(32, (5, 5), activation='relu')
    board_conv3_bn_layer = BatchNormalization()
    board_conv3_activation_layer = Activation('relu')
    board_conv3_attention_layer = Attention()

    board_flat_layer = Flatten()
    board_pre_dense_layer = Dense(32, activation='relu')

    # connecting
    board_conv1_out = board_conv1_layer(input_board)
    board_conv1_bn_out = board_conv1_bn_layer(board_conv1_out)
    board_conv1_activation_out = board_conv1_activation_layer(board_conv1_bn_out)

    board_conv2_out = board_conv2_layer(board_conv1_activation_out)
    board_conv2_bn_out = board_conv2_bn_layer(board_conv2_out)
    board_conv2_activation_out = board_conv2_activation_layer(board_conv2_bn_out)
    board_conv2_attention_out = board_conv2_attention_layer([board_conv2_activation_out, board_conv2_activation_out])

    board_conv3_out = board_conv3_layer(board_conv2_attention_out)
    board_conv3_bn_out = board_conv3_bn_layer(board_conv3_out)
    board_conv3_activation_out = board_conv3_activation_layer(board_conv3_bn_out)
    board_conv3_attention_out = board_conv3_attention_layer([board_conv3_activation_out, board_conv3_activation_out])

    board_flat_out = board_flat_layer(board_conv3_attention_out)
    board_pre_dense_out = board_pre_dense_layer(board_flat_out)

    # append meta
    concat_meta = concatenate([board_pre_dense_out, input_meta])

    # combined arch
    combined1_layer = Dense(32, activation = 'relu')
    combined2_layer = Dense(16, activation = 'relu')
    output_layer = Dense(1, activation='linear')

    # connecting
    combined1_out = combined1_layer(concat_meta)
    combined2_out = combined2_layer(combined1_out)
    output = output_layer(combined2_out)

    # build the model
    model = Model(inputs=[input_board, input_meta], outputs=output)
    return model

def get_model_002():
    input_board = Input(shape=(8, 8, 12))
    input_meta = Input(shape=(5,))

    # layer arch
    board_conv1_layer = Conv2D(32, (3, 3))
    board_conv1_pooling_layer = MaxPooling2D((2, 2), padding='same')
    board_conv1_bn_layer = BatchNormalization()
    board_conv1_activation_layer = Activation('relu')

    board_conv2_layer = Conv2D(64, (3, 3))
    board_conv2_pooling_layer = MaxPooling2D((2, 2), padding='same')
    board_conv2_bn_layer = BatchNormalization()
    board_conv2_attention_layer = Attention()
    board_conv2_activation_layer = Activation('relu')
    board_conv2_global_pooling_layer = GlobalAveragePooling2D()

    board_flat_layer = Flatten()

    # connecting
    board_conv1_out = board_conv1_layer(input_board)
    board_conv1_pooling_out = board_conv1_pooling_layer(board_conv1_out)
    board_conv1_bn_out = board_conv1_bn_layer(board_conv1_pooling_out)
    board_conv1_activation_out = board_conv1_activation_layer(board_conv1_bn_out)

    board_conv2_out = board_conv2_layer(board_conv1_activation_out)
    board_conv2_pooling_out = board_conv2_pooling_layer(board_conv2_out)
    board_conv2_bn_out = board_conv2_bn_layer(board_conv2_pooling_out)
    board_conv2_attention_out = board_conv2_attention_layer([board_conv2_bn_out, board_conv2_bn_out])
    board_conv2_activation_out = board_conv2_activation_layer(board_conv2_attention_out)
    board_conv2_global_pooling_out = board_conv2_global_pooling_layer(board_conv2_activation_out)

    board_flat_out = board_flat_layer(board_conv2_global_pooling_out)

    # append meta
    concat_meta = concatenate([board_flat_out, input_meta])

    # combined arch
    combined1_layer = Dense(128, activation = 'relu')
    output_layer = Dense(1, activation='linear')

    # connecting
    combined1_out = combined1_layer(concat_meta)
    output = output_layer(combined1_out)

    # build the model
    model = Model(inputs=[input_board, input_meta], outputs=output)
    return model

def get_value_model_001():
    input_board = tf.keras.layers.Input(shape=(8, 8, 6))
    input_meta = tf.keras.layers.Input(shape=(6,))
    cnn1 = tf.keras.layers.Conv2D(64, (4, 4), padding='same', activation='relu')
    bn1 = tf.keras.layers.BatchNormalization()
    cnn2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    bn2 = tf.keras.layers.BatchNormalization()
    f1 = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(128, activation='relu')
    d2 = tf.keras.layers.Dense(1, activation='tanh')

    cnn1_out = cnn1(input_board)
    bn1_out = bn1(cnn1_out)
    cnn2_out = cnn2(bn1_out)
    bn2_out = bn2(cnn2_out)
    f1_out = f1(bn2_out)
    d1_out = d1(concatenate([f1_out, input_meta]))
    d2_out = d2(d1_out)

    model = Model(inputs=[input_board, input_meta], outputs=d2_out)
    return model

def get_value_model_002():
    input_board = tf.keras.layers.Input(shape=(8, 8, 6))
    input_meta = tf.keras.layers.Input(shape=(6,))
    cnn1 = tf.keras.layers.Conv2D(64, (4, 4), padding='same', activation='relu')
    bn1 = tf.keras.layers.BatchNormalization()
    cnn2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    bn2 = tf.keras.layers.BatchNormalization()
    f1 = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(128, activation='relu')
    d2 = tf.keras.layers.Dense(1, activation='tanh')

    cnn1_out = cnn1(input_board)
    bn1_out = bn1(cnn1_out)
    cnn2_out = cnn2(bn1_out)
    bn2_out = bn2(cnn2_out)
    f1_out = f1(bn2_out)
    d1_out = d1(concatenate([f1_out, input_meta]))
    d2_out = d2(d1_out)

    model = Model(inputs=[input_board, input_meta], outputs=d2_out)
    return model


def get_value_model_003():
    input_board = tf.keras.layers.Input(shape=(8, 8, 7))
    input_meta = tf.keras.layers.Input(shape=(7,))

    cnn1 = tf.keras.layers.Conv2D(64, (4, 4), padding='same')
    att1 = tf.keras.layers.Attention()
    bn1 = tf.keras.layers.BatchNormalization()
    act1 = tf.keras.layers.Activation('relu')

    cnn2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
    att2 = tf.keras.layers.Attention()
    bn2 = tf.keras.layers.BatchNormalization()
    act2 = tf.keras.layers.Activation('relu')

    cnn3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
    att3 = tf.keras.layers.Attention()
    bn3 = tf.keras.layers.BatchNormalization()
    act3 = tf.keras.layers.Activation('relu')

    cnn4 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
    att4 = tf.keras.layers.Attention()
    bn4 = tf.keras.layers.BatchNormalization()
    act4 = tf.keras.layers.Activation('relu')

    f1 = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(128, activation='relu')
    bn5 = tf.keras.layers.BatchNormalization()
    d2 = tf.keras.layers.Dense(1, activation='tanh')

    cnn1_out = cnn1(input_board)
    att1_out = att1([cnn1_out, cnn1_out])
    bn1_out = bn1(att1_out)
    act1_out = act1(bn1_out)

    cnn2_out = cnn2(act1_out)
    att2_out = att2([cnn2_out, cnn2_out])
    bn2_out = bn2(att2_out)
    act2_out = act2(bn2_out)

    cnn3_out = cnn3(act2_out)
    att3_out = att3([cnn3_out, cnn3_out])
    bn3_out = bn3(att3_out)
    act3_out = act3(bn3_out)

    cnn4_out = cnn4(act3_out)
    att4_out = att4([cnn4_out, cnn4_out])
    bn4_out = bn4(att4_out)
    act4_out = act4(bn4_out)

    f1_out = f1(act4_out)
    d1_out = d1(concatenate([f1_out, input_meta]))
    bn5_out = bn5(d1_out)
    d2_out = d2(bn5_out)

    model = Model(inputs=[input_board, input_meta], outputs=d2_out)
    return model

def get_policy_model_001():
    input_board = tf.keras.layers.Input(shape=(8, 8, 6))

    cnn1 = tf.keras.layers.Conv2D(64, (4, 4), padding='same')
    att1 = tf.keras.layers.Attention()
    bn1 = tf.keras.layers.BatchNormalization()
    act1 = tf.keras.layers.Activation('relu')

    cnn2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
    att2 = tf.keras.layers.Attention()
    bn2 = tf.keras.layers.BatchNormalization()
    act2 = tf.keras.layers.Activation('relu')

    f1 = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(128, activation='relu')
    d2 = tf.keras.layers.Dense(64*64, activation='softmax')

    cnn1_out = cnn1(input_board)
    att1_out = att1([cnn1_out, cnn1_out])
    bn1_out = bn1(att1_out)
    act1_out = act1(bn1_out)
    
    cnn2_out = cnn2(act1_out)
    att2_out = att2([cnn2_out, cnn2_out])
    bn2_out = bn2(att2_out)
    act2_out = act2(bn2_out)

    f1_out = f1(act2_out)
    d1_out = d1(f1_out)
    d2_out = d2(d1_out)

    model = Model(inputs=input_board, outputs=d2_out)
    return model

def get_policy_model_003():
    input_board = tf.keras.layers.Input(shape=(8, 8, 7))
    cnn1 = tf.keras.layers.Conv2D(64, (4, 4), padding='same', activation='relu')
    bn1 = tf.keras.layers.BatchNormalization()
    cnn2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    bn2 = tf.keras.layers.BatchNormalization()
    f1 = tf.keras.layers.Flatten()
    d1 = tf.keras.layers.Dense(128, activation='relu')
    d2 = tf.keras.layers.Dense(64*64, activation='softmax')

    cnn1_out = cnn1(input_board)
    bn1_out = bn1(cnn1_out)
    cnn2_out = cnn2(bn1_out)
    bn2_out = bn2(cnn2_out)
    f1_out = f1(bn2_out)
    d1_out = d1(f1_out)
    d2_out = d2(d1_out)

    model = Model(inputs=input_board, outputs=d2_out)
    return model

def compile_regression_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.MeanSquaredError()
    )

def compile_experimental_classifier_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalCrossentropy()
    )
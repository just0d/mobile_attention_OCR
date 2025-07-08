import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Bidirectional, LSTM, Concatenate, Permute,
    Reshape, Dropout, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import regularizers

# Use the same global variables you defined in utils.py or in your main code:
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'\" "
max_str_len = 2000
num_of_characters = len(alphabets) + 2
START_TOKEN = len(alphabets)
END_TOKEN = len(alphabets) + 1
PAD_TOKEN = 0

def build_academic_mobilenet_model(input_shape=(64, 256, 1)):
    """Build MobileNetV3 + BiLSTM + Attention OCR model"""

    # Input layer
    inp = Input(shape=input_shape, name='input')
    x = Concatenate(axis=-1)([inp, inp, inp])  # Convert grayscale to 3-channel

    # MobileNetV3Small backbone
    mobilenet = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        input_shape=(64, 256, 3),
        alpha=1.0,
        minimalistic=False,
        include_preprocessing=False
    )

    # Feature processing
    features = mobilenet.output
    features = Permute((2, 1, 3))(features)
    features = Reshape((8, -1))(features)

    # BiLSTM encoder with regularization
    encoder_lstm = Bidirectional(
        LSTM(
            256, return_sequences=True, dropout=0.1,
            recurrent_dropout=0.05,
            kernel_regularizer=regularizers.L2(1e-6),
            recurrent_regularizer=regularizers.L1(1e-6),
            bias_regularizer=regularizers.L2(1e-4)
        )
    )(features)

    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.1
    )(encoder_lstm, encoder_lstm)

    encoder_output = Add()([encoder_lstm, attention_output])
    encoder_output = LayerNormalization()(encoder_output)

    # Final encoder BiLSTM
    encoder_final = Bidirectional(
        LSTM(
            128, dropout=0.2, recurrent_dropout=0.1,
            kernel_regularizer=regularizers.L2(1e-4),
            recurrent_regularizer=regularizers.L1(1e-5),
            bias_regularizer=regularizers.L2(1e-4)
        )
    )(encoder_output)

    # Decoder inputs
    decoder_inputs = Input(shape=(max_str_len,), name='decoder_inputs')

    # Embedding layer with regularization
    decoder_embedding = tf.keras.layers.Embedding(
        num_of_characters,
        64,
        mask_zero=False,
        embeddings_regularizer=regularizers.L2(1e-4)
    )(decoder_inputs)
    decoder_embedding = Dropout(0.3)(decoder_embedding)

    # Context vector repeated for each timestep
    context = tf.keras.layers.RepeatVector(max_str_len)(encoder_final)

    # Cross-attention layer
    cross_attention = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.1
    )(decoder_embedding, encoder_output)

    # Combine decoder inputs
    decoder_combined = Concatenate(axis=-1)([
        decoder_embedding,
        context,
        cross_attention
    ])

    # Decoder BiLSTM layers
    decoder_lstm1 = LSTM(
        128,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=regularizers.L2(1e-4),
        recurrent_regularizer=regularizers.L1(1e-5),
        bias_regularizer=regularizers.L2(1e-4)
    )(decoder_combined)

    decoder_lstm2 = LSTM(
        96,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.2,
        kernel_regularizer=regularizers.L2(1e-4),
        recurrent_regularizer=regularizers.L1(1e-5),
        bias_regularizer=regularizers.L2(1e-4)
    )(decoder_lstm1)

    # Output dense layer with softmax activation
    outputs = Dense(
        num_of_characters,
        activation='softmax',
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4)
    )(decoder_lstm2)

    model = Model([inp, decoder_inputs], outputs, name='Academic_MobileNetV3_BiLSTM_Attention')

    return model
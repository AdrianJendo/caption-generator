from pickle import load
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from helpers import (
    dataset_text,
    load_photos,
    load_descriptions,
    load_features,
    dict_to_list,
)

filename = dataset_text + "/Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename)
train_descriptions = load_descriptions("./data/descriptions.txt", train_imgs)
train_features = load_features(train_imgs)
tokenizer = load(open("tokenizer.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1
features = load(open("features.p", "rb"))
print("Vocab size:", vocab_size)

# define the captioning model
def define_model(vocab_size, max_length):

    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merging both models
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    # summarize model
    print(model.summary())
    plot_model(model, to_file="model.png", show_shapes=True)

    return model


# train our model
# create input-output sequence pairs from the image description.

# data generator, used by model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            # retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(
                tokenizer, max_length, description_list, feature
            )
            yield [[input_image, input_sequence], output_word]


def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


def get_max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


max_length = get_max_length(train_descriptions)
print(max_length)

features = load(open("features.p", "rb"))

# You can check the shape of the input and output for your model
[a, b], c = next(data_generator(train_descriptions, features, tokenizer, max_length))
print(a.shape, b.shape, c.shape)


print("Dataset: ", len(train_imgs))
print("Descriptions: train=", len(train_descriptions))
print("Photos: train=", len(train_features))
print("Vocabulary Size:", vocab_size)
print("Description Length: ", max_length)
# model = define_model(vocab_size, max_length)
model = load_model("models/model_10.h5")  # stopped training here
epochs = 15
steps = len(train_descriptions)
for i in range(11, epochs):
    generator = data_generator(
        train_descriptions, train_features, tokenizer, max_length
    )
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")

from pickle import dump
from tensorflow.keras.preprocessing.text import Tokenizer

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


def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open("tokenizer.p", "wb"))
vocab_size = len(tokenizer.word_index) + 1
print("Vocab size:", vocab_size)

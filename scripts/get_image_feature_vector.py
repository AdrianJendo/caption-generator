import numpy as np
from PIL import Image
import os
from pickle import dump
import numpy as np
from tensorflow.keras.applications.xception import Xception
from tqdm import tqdm
from helpers import dataset_images

# extract feature vector for all images and store in .pickle file
def extract_features(directory):
    model = Xception(include_top=False, pooling="avg")
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        # image = preprocess_input(image)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        features[img] = feature
    return features


# 2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p", "wb"))

from pickle import load

dataset_images = "./data/Flicker8k_Dataset"
dataset_text = "./data/Flickr8k_text"

# Load text file
def load_doc(filename):
    # Read only
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text


# Get all images with captions
def get_descriptions(filename):
    file = load_doc(filename)
    captions = file.split("\n")
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split("\t")
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions


def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return photos


# load the cleaned descriptions
def load_descriptions(filename, photos):
    # loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = "<start> " + " ".join(image_caption) + " <end>"
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    # loading all features
    all_features = load(open("features.p", "rb"))
    # selecting only needed features
    features = {k: all_features[k] for k in photos}
    return features


# converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

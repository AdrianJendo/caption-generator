import string
from helpers import dataset_text, load_doc


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


# Data cleaning - lower case, no punctiation, remove words with numbers
def clean_text(captions):
    # Create dict !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ in ascii as the key
    table = str.maketrans("", "", string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split()

            # converts to lowercase
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            # remove hanging 's and a
            # desc = [word for word in desc if (len(word) > 1)]
            # remove tokens with numbers in them
            desc = [
                word for word in desc if (word.isalpha())
            ]  # True if all chars in string are from a-z

            # convert back to string
            img_caption = " ".join(desc)
            # add to dict
            captions[img][i] = img_caption
    return captions


def text_vocabulary(descriptions):
    # build vocabulary of all words in dataset
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab


# All descriptions in one file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + "\t" + desc)
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


# load the data file
# Format of descriptions is {img_name : [list of 5 captions]}
descriptions = get_descriptions(dataset_text + "/Flickr8k.token.txt")
print("Length of descriptions =", len(descriptions))

# cleaning the descriptions
clean_descriptions = clean_text(descriptions)

# building vocabulary
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))

# saving each description to file
save_descriptions(clean_descriptions, "./data/descriptions.txt")

dataset_images = "./data/Flicker8k_Dataset"
dataset_text = "./data/Flickr8k_text"

# Load text file
def load_doc(filename):
    # Read only
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text

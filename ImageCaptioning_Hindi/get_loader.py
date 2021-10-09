import os  # when loading file paths
from pathlib import Path
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms


# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self, freq_threshold):
        """
            input : frequency threshold (ignore if not repeat appear enough time)
            output : initialize dictionary for conversion 
        """
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}     # index to string
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}     # string to index
        self.freq_threshold = freq_threshold      

    def __len__(self):
        """
            output: return length of vocabulary
        """
        return len(self.itos)

    @staticmethod       # do not require class object creation
    def tokenizer_eng(text):
        """
            tokenize the sentences into words
            input : string of sentence
            output : list of words
        """
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
        
    def build_vocabulary(self, sentence_list):
        """
            input : list of sentences
        """
        frequencies = {}  # of each word 
        idx = 4           # itos or stoi length
        
        # for each sentence
        for sentence in sentence_list:
            # for each word
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:    # add word to dict
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1     # increment frequency

                # if frequency reaches to threshold
                if frequencies[word] == self.freq_threshold:     
                    self.stoi[word] = idx    # add words
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        """
            input : sentence
        """
        # tokenize the sentence
        tokenized_text = self.tokenizer_eng(text)

        # convert to int using stoi
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=1):  # previous 5
        """
            input : images path, captions file path, transfrom if have any, frequency threshold
            ouput: initialize images, captions dataframe, and vocabulary
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, encoding = 'utf-8')  # change it so that it can read with spaces
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize vocabulary instance
        self.vocab = Vocabulary(freq_threshold)
        
        # create a vocabulary
        self.vocab.build_vocabulary(self.captions.tolist())    # create a list of all captions

    def __len__(self):
        """
            return length of captions dataframe
        """
        return len(self.df)

    def __getitem__(self, index):
        """
            input: index of image
            output:
        """
        # get the caption and image id
        caption = self.captions[index]
        img_id = self.imgs[index]

        # in hindi captions, image id is given without .jpg so added .jpg
        img_id += '.jpg'
        
        # open the image
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        
        # apply transform if given
        if self.transform is not None:
            img = self.transform(img)
        
        # convert words into integers
        numericalized_caption = [self.vocab.stoi["<SOS>"]]          # start
        numericalized_caption += self.vocab.numericalize(caption)   # words
        numericalized_caption.append(self.vocab.stoi["<EOS>"])      # and
        
#         print(numericalized_caption)
        
        # return the image and numericalized caption
        return img, torch.tensor(numericalized_caption)


class MyCollate:
    # pad index
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    # get list of lists of examples
    def __call__(self, batch):
        # get batch of images
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)    # concatenate accoss dim=0
        
        # get batch of captions
        targets = [item[1] for item in batch]
        
        # pad the captions batch
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets


# function to get dataloader
def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    # get dataset
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    
    # pad the indexes
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
#     print(dataset.vocab.stoi, dataset.vocab.itos)
    
    # create a pytorch dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx), # collate function for dataloader
    )

    return loader, dataset


if __name__ == "__main__":
    
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), 
         transforms.ToTensor(),]
    )
    
    # get the dataloader and dataset
    loader, dataset = get_loader(
        "../Data/test_examples/images/", "../Data/test_examples/captions_hindi.txt", transform=transform
    )

    # print image shapes and captions shape
    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)     # torch.size([32, 3, 224, 224])  like (batch_size, channels, h, w)
        print(captions.shape)   # torch.size([26, 32])  like (caption_length, batch_size)
import os, sys, re, pickle, json
import numpy as np

import torch
import torch.utils.data as data

from utils import Vocabulary

def set_rnd_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_number(w):    
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

class EvalDataLoader(data.Dataset):
    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        self.captions = list()
        self.labels = list()
        self.spans = list()
        self.tags = list()
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f:
            for line in f:
                (caption, span, label, tag) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                self.captions.append(caption)
                self.labels.append(label)
                self.spans.append(span)
                self.tags.append(tag)
        self.length = len(self.captions)

    def __getitem__(self, index):
        caption = [self.vocab(token) for token in self.captions[index]]
        caption = torch.tensor(caption)
        label = self.labels[index]
        span = self.spans[index]
        tag = self.tags[index]
        return caption, label, span, tag, index 

    def __len__(self):
        return self.length

def collate_fun_eval(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    zipped_data = list(zip(*data))
    captions, labels, spans, tags, ids = zipped_data
    max_len = max([len(caption) for caption in captions]) 
    targets = torch.zeros(len(captions), max_len).long()
    lengths = [len(cap) for cap in captions]
    for i, cap_len in enumerate(lengths):
        targets[i, : cap_len] = captions[i][: cap_len]
    return targets, lengths, spans, labels, tags, ids 

def eval_data_iter(data_path, data_split, vocab, batch_size=128):
    data = EvalDataLoader(data_path, data_split, vocab)
    data_loader = torch.utils.data.DataLoader(
                    dataset=data, 
                    batch_size=batch_size, 
                    shuffle=False,
                    sampler=None,
                    pin_memory=True, 
                    collate_fn=collate_fun_eval
    )
    return data_loader

class SortedRandomSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class SortedSequentialSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.data_source._shuffle()
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class DataLoader(data.Dataset):
    """ sentences and their spans """
    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab
        # captions
        self.captions = list()
        self.spans = list()
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f:
            for line in f:
                (caption, span) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if len(caption) < 2 or len(caption) > 70:
                    continue
                self.captions.append(caption)
                self.spans.append(span)
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 1, img_dim))
        
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

        # sort captions by length by default
        self.image_idxes = np.repeat(range(int(self.length / self.im_div)), self.im_div)

    def _shuffle(self):
        indice = torch.randperm(self.length).tolist() 
        indice = sorted(indice, key=lambda k: len(self.captions[k]))
        self.spans = [self.spans[k] for k in indice]
        self.captions = [self.captions[k] for k in indice]
        self.image_idxes = self.image_idxes[indice]

    def __getitem__(self, index):
        # image
        #img_id = index  // self.im_div
        img_id = self.image_idxes[index]
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [self.vocab(token) for token in self.captions[index]]
        caption = torch.tensor(caption)
        span = self.spans[index]
        span = torch.tensor(span)
        return image, caption, index, img_id, span

    def __len__(self):
        return self.length

def collate_fun(data):
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, img_ids, spans = zipped_data
    images = torch.stack(images, 0)
    max_len = max([len(caption) for caption in captions]) 
    targets = torch.zeros(len(captions), max_len).long()
    lengths = [len(cap) for cap in captions]
    indices = torch.zeros(len(captions), max_len, 2).long()
    for i, cap in enumerate(captions):
        cap_len = len(cap)
        targets[i, : cap_len] = cap[: cap_len]
        indices[i, : cap_len - 1, :] = spans[i]
    return images, targets, lengths, ids, indices 

def get_data_loader(data_path, data_split, vocab, 
                    batch_size=128, 
                    shuffle=True, 
                    nworker=2, 
                    loadimg=True, 
                    img_dim=2048,
                    sampler=None):
    dset = DataLoader(data_path, data_split, vocab, loadimg, img_dim)
    if sampler:
        sampler = SortedRandomSampler(dset)
    data_loader = torch.utils.data.DataLoader(
                    dataset=dset, 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True, 
                    collate_fn=collate_fun
    )
    return data_loader

def get_train_iters(data_path, vocab, batch_size, nworker):
    train_loader = get_data_loader(
        data_path, 'ptb-train', vocab, 
        batch_size=batch_size, shuffle=False, nworker=nworker, sampler=True, loadimg=False
    )
    val_loader = get_data_loader(
        data_path, 'ptb-val', vocab, 
        batch_size=batch_size, shuffle=False, nworker=nworker, sampler=None, loadimg=False
    )
    return train_loader, val_loader

def get_eval_iter(data_path, split_name, vocab, batch_size, 
                  nworker=2, 
                  shuffle=False,
                  loadimg=False, 
                  img_dim=2048,
                  sampler=None):
    eval_loader = get_data_loader(data_path, split_name, vocab, 
                  batch_size=batch_size, 
                  shuffle=shuffle,  
                  nworker=nworker, 
                  loadimg=loadimg, 
                  img_dim=img_dim,
                  sampler=sampler
    )
    return eval_loader


def test_data_loader():
    data_path = "/home/s1847450/data/vsyntax/mscoco/"
    vocab = pickle.load(open(data_path + "coco.dict.pkl", 'rb'))

    batch_size = 3 
    nworker = 2
    
    train_data_iter, val_data_iter = get_train_iters(
        data_path, vocab, batch_size, nworker
    )

    """
    for images, targets, lengths, ids, spans in train_data_iter:
        print(images.size(), ids, lengths)
        print(spans)
        break
    for images, targets, lengths, ids, spans in val_data_iter:
        print(images.size(), ids, lengths)
        print(spans)
        break
    """

    shuffle = False 
    data_split = "val"
    data_iter = get_eval_iter(data_path, data_split, vocab, 
        batch_size=batch_size, shuffle=shuffle, nworker=nworker)
    for images, targets, lengths, ids, spans in data_iter:
        print(images.size(), ids, lengths)
        print(spans)
        break

def test_sorted_data_loader():
    data_path = "/home/s1847450/data/vsyntax/mscoco/"
    vocab = pickle.load(open(data_path + "coco.dict.pkl", 'rb'))

    batch_size = 5 
    nworker = 2
    
    shuffle = False 
    data_split = "val"
    data_iter = get_eval_iter(data_path, data_split, vocab, 
        batch_size=batch_size, shuffle=shuffle, nworker=nworker, sampler=True)
    for i in range(3):
        for j, (images, targets, lengths, ids, spans) in enumerate(data_iter):
            print(images.size(), ids, lengths)
            print(targets)
            print(spans)
            if j == 1:
                break

def test_eval_data_loader():
    data_path = "/home/s1847450/data/vsyntax/mscoco/"
    vocab = pickle.load(open(data_path + "coco.dict.pkl", 'rb'))

    batch_size = 3 
    data_split = "val_gold"
    data_iter = eval_data_iter(data_path, data_split, vocab, batch_size=batch_size)
    for i in range(1):
        for j, (captions, lengths, spans, labels, tags, ids) in enumerate(data_iter):
            print("ids: ", ids)
            print("lengths: ", lengths)
            print("captions:\n", captions)
            print("labels:\n", labels)
            print("spans:\n", spans)
            print("tags:\n", tags)
            if j == 1:
                break

if __name__ == '__main__':
    #test_data_loader()
    #test_sorted_data_loader()
    test_eval_data_loader()
    pass

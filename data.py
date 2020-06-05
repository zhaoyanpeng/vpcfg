import os, sys, re, pickle, json
import numpy as np
import random

from gensim.test.utils import datapath
from gensim.models import KeyedVectors

import torch
import torch.utils.data as data

from utils import Vocabulary

TXT_IMG_DIVISOR=1
TXT_MAX_LENGTH=45

def set_constant(visual_mode, max_length):
    global TXT_IMG_DIVISOR, TXT_MAX_LENGTH
    TXT_IMG_DIVISOR = 1 if not visual_mode else 5 
    TXT_MAX_LENGTH = max_length
    #print(TXT_IMG_DIVISOR, TXT_MAX_LENGTH)

def set_rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_number(w):    
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, vocab, w2vec_file, dim=-1):
        super(PartiallyFixedEmbedding, self).__init__()
        nword = len(vocab)
        model = KeyedVectors.load_word2vec_format(datapath(w2vec_file), binary=False)
        masks = [1 if vocab.idx2word[k] in model.vocab else 0 for k in range(nword)]
        idx2fixed = [k for k in range(nword) if masks[k]]
        idx2tuned = [k for k in range(nword) if not masks[k]]
        arranged_idx = idx2fixed + idx2tuned
        idx_mapping = {idx: real_idx for real_idx, idx in enumerate(arranged_idx)}
        self.idx_mapping = idx_mapping
        self.n_fixed = sum(masks)
        n_tuned = nword - self.n_fixed

        weight = torch.empty(nword, model.vector_size)
        for k, idx in vocab.word2idx.items():
            real_idx = idx_mapping[idx]
            if k in model.vocab:
                weight[real_idx] = torch.tensor(model[k])

        self.tuned_weight = torch.nn.Parameter(torch.empty(n_tuned, model.vector_size)) 
        torch.nn.init.kaiming_uniform_(self.tuned_weight)
        weight[self.n_fixed:] = self.tuned_weight
        self.register_buffer("weight", weight)
         
        dim = dim - model.vector_size if dim > model.vector_size else 0 
        self.tuned_vector = torch.nn.Parameter(torch.empty(nword, dim))
        if dim > 0: 
            torch.nn.init.kaiming_uniform_(self.tuned_vector)
        del model

    def reindex(self, X):
        return X.clone().cpu().apply_(self.idx_mapping.get)

    def forward(self, X):
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        weight = torch.cat([self.weight, self.tuned_vector], -1)
        #X = X.clone().cpu().apply_(self.idx_mapping.get) # only work on cpus
        X = X.clone().cpu().apply_(self.idx_mapping.get).cuda() 
        #print(X, weight.device, self.weight.device, self.tuned_vector.device)
        return torch.nn.functional.embedding(X, weight, None, None, 2.0, False, False)

class EvalDataLoader(data.Dataset):
    def __init__(self, data_path, data_split, vocab, min_length=1, max_length=100):
        self.vocab = vocab
        self.captions = list()
        self.labels = list()
        self.spans = list()
        self.tags = list()
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f:
            for line in f:
                (caption, span, label, tag) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if len(caption) < min_length or len(caption) > max_length:
                    continue
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

def eval_data_iter(data_path, data_split, vocab, batch_size=128, 
                   min_length=1, max_length=100):
    data = EvalDataLoader(data_path, data_split, vocab, 
        min_length=min_length, max_length=max_length)
    data_loader = torch.utils.data.DataLoader(
                    dataset=data, 
                    batch_size=batch_size, 
                    shuffle=False,
                    sampler=None,
                    pin_memory=True, 
                    collate_fn=collate_fun_eval
    )
    return data_loader

class SortedBlockSampler(data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        all_sample = len(self.data_source)
        batch_size = data_source.batch_size
        nblock = all_sample // batch_size 
        residue = all_sample % batch_size
        nsample = all_sample - residue
        # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        self.groups = np.array_split(range(nsample), nblock)
        self.strip_last = False
        if residue > 0:
            self.strip_last = True
            block = np.array(range(nsample, all_sample))
            self.groups.append(block)

    def __iter__(self):
        self.data_source._shuffle()
        end = -1 if self.strip_last else len(self.groups)
        groups = self.groups[:end]
        #random.shuffle(groups) 
        indice = torch.randperm(len(groups)).tolist() 
        groups = [groups[k] for k in indice]
        if self.strip_last:
            groups.append(self.groups[-1])
        indice = list()
        for i, group in enumerate(groups):
            indice.extend(group)
            #print(i, group)
        assert len(indice) == len(self.data_source)
        return iter(indice)

    def __len__(self):
        return len(self.data_source)

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
    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048, batch_size=1):
        self.batch_size = batch_size
        self.vocab = vocab
        self.captions = list()
        self.spans = list()
        max_length = TXT_MAX_LENGTH 
        if TXT_MAX_LENGTH < 1000 and "val" in data_split:
            max_length = 50 # mem issue
        indexes, removed, idx = list(), list(), -1 
        with open(os.path.join(data_path, f'{data_split}_caps.json'), 'r') as f:
            for line in f:
                idx += 1
                (caption, span) = json.loads(line)
                caption = [clean_number(w) for w in caption.strip().lower().split()]
                if TXT_MAX_LENGTH < 1000 and (len(caption) < 2 or len(caption) > max_length):
                    removed.append((idx, len(caption))) 
                    self.captions.append(-1)
                    self.spans.append(-1)
                    indexes.append(-1)
                    continue
                self.captions.append(caption)
                self.spans.append(span)
                indexes.append(idx)
        self.length = len(self.captions)
        self.im_div = TXT_IMG_DIVISOR

        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // TXT_IMG_DIVISOR, img_dim))
        
        fname = os.path.join(data_path, f'{data_split}_caps.json')
        #print(len(indexes), len(removed), idx, self.length, self.im_div, load_img, removed) 
        if len(removed) > 0: # remove -1
            assert len(indexes) == self.images.shape[0] * TXT_IMG_DIVISOR
            groups = np.array_split(indexes, self.images.shape[0])
            indice, image_idxes = list(), list() 
            for idx, group in enumerate(groups):
                if -1 not in group:
                    indice.extend(group)
                    image_idxes.append(idx)
                else:
                    #print(idx, group)
                    pass
            self.spans = [self.spans[k] for k in indice]
            self.captions = [self.captions[k] for k in indice]
            self.images = self.images[image_idxes]
            self.length = len(self.captions)
            assert self.length == self.images.shape[0] * TXT_IMG_DIVISOR
        #print(self.length, len(self.spans), self.images.shape[0]) 
        
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
    dset = DataLoader(data_path, data_split, vocab, loadimg, img_dim, batch_size)
    if sampler:
        model = SortedRandomSampler
        if not isinstance(sampler, bool) and issubclass(sampler, data.Sampler):
            model = sampler
        #sampler = SortedRandomSampler(dset)
        sampler = model(dset)
    data_loader = torch.utils.data.DataLoader(
                    dataset=dset, 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    sampler=sampler,
                    pin_memory=True, 
                    collate_fn=collate_fun
    )
    return data_loader

def get_train_iters(data_path, prefix, vocab, batch_size, nworker, loadimg=True, sampler=True):
    train_loader = get_data_loader(
        data_path, prefix + 'train', vocab, 
        batch_size=batch_size, shuffle=False, nworker=nworker, sampler=sampler, loadimg=loadimg
    )
    val_loader = get_data_loader(
        data_path, prefix + 'val', vocab, 
        batch_size=batch_size, shuffle=False, nworker=nworker, sampler=None, loadimg=loadimg
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

def test_embedder():
    data_path = "/home/s1847450/data/vsyntax/mstree/"
    vocab = pickle.load(open(data_path + "msp.dict.pkl", 'rb'))
    w2vec_file = data_path + "msp.dict.6B.200d.vec"
    
    dim = 200
    enc_emb = PartiallyFixedEmbedding(vocab, w2vec_file, dim)
    enc_emb = enc_emb.cuda()
    model = KeyedVectors.load_word2vec_format(datapath(w2vec_file), binary=False)
    
    word = ["seas", "plane", "clasps", "stylist"]
    widx = [vocab.word2idx[w] for w in word]
    widx = torch.tensor(widx).long()
    
    vec0 = model[word]
    vec1 = enc_emb(widx)
    
    print(vec0.size())
    print(vec1.size())
    assert vec0.tolist() == vec1.tolist()
    
def test_block_data_loader():
    data_path = "/home/s1847450/data/vsyntax/treebk/"
    vocab = pickle.load(open(data_path + "ptb.dict.pkl", 'rb'))
    
    seed = 6768 
    set_rnd_seed(seed)

    batch_size = 10 
    data_split = "ptb-val"
    sampler = SortedBlockSampler
    #sampler = None 
    data_iter = get_eval_iter(data_path, data_split, vocab, batch_size=batch_size, sampler=sampler)
    for i in range(1):
        for j, (images, targets, lengths, ids, spans) in enumerate(data_iter):
            print(j, ids, "lengths: ", lengths)
            if j == 10:
                #break
                pass

if __name__ == '__main__':
    #test_data_loader()
    #test_sorted_data_loader()
    #test_eval_data_loader()
    #test_embedder()
    test_block_data_loader()
    pass

import time
import numpy as np
from collections import OrderedDict

import torch

import utils


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s


def encode_data(model, data_loader, log_step=10, logging=print, vocab=None, stage='dev'):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    # stats
    n_word, n_sent = 0, 0
    total_ll, total_kl, total_h = 0., 0., 0.
    sent_f1, corpus_f1 = [], [0., 0., 0.] 

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        # compute the embeddings
        bsize = images.size(0) #cum_reward.shape[0]
        img_emb, cap_span_features, left_span_features, right_span_features, \
            word_embs, tree_indices, probs, span_bounds, nll, kl, bc, h, span_margs, \
                argmax_spans, trees, lprobs = model.forward_encoder(
            images, captions, lengths, spans, require_grad=False
        )

        # output sampled trees
        if (not logged) or (stage == 'test'):
            logged = True
            sample_num = 0
            if stage == 'dev' and tree_indices is not None:
                sample_num = 5
            for j in range(sample_num):
                pass

        mstep = (lengths * (lengths - 1) / 2).int() # (b, NT, dim) 
        cap_feats = torch.cat(
            [cap_span_features[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0
        ) 
        span_marg = torch.softmax(
            torch.cat([span_margs[j][k - 1].unsqueeze(0) for j, k in enumerate(mstep)], dim=0), -1
        )
        cap_emb = torch.bmm(span_marg.unsqueeze(-2),  cap_feats).squeeze(-2)
        cap_emb = utils.l2norm(cap_emb)

        #print(nll, kl, lengths)
        #print(captions)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        total_h += h.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        # F1
        bsize = img_emb.shape[0]
        for b in range(bsize):
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)] 
            gold_set = set(gold[:-1])

            tp, fp, fn = utils.get_stats(pred_set, gold_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            
            overlap = pred_set.intersection(gold_set)
            prec = float(len(overlap)) / (len(pred_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_set) + 1e-8)
            
            if len(gold_set) == 0:
                reca = 1. 
                if len(pred_set) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

        #if i >= 50: break

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    h = total_h / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}, H: {:.4f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, h, corpus_f1 * 100, sent_f1 * 100
    )
    logging(info)
    return img_embs, cap_embs, ppl_elbo, sent_f1 * 100 

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def validate(opt, val_loader, model, vocab, logger):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, val_ppl, val_f1 = encode_data(
        model, val_loader, opt.log_step, logger.info, vocab)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure='cosine')
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure='cosine')
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return val_ppl
    #return currscore

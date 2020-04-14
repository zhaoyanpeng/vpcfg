import os
import time, pickle
import numpy as np

import torch
import utils

from collections import OrderedDict

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
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl, total_bc, total_h = 0., 0., 0., 0.

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()

        # compute the embeddings
        bsize = images.size(0) #cum_reward.shape[0]
        model_output = model.forward_emb(
            images, captions, lengths, spans, volatile=True
        )
        img_emb, cap_span_features, left_span_features, right_span_features, \
            word_embs, tree_indices, all_probs, span_bounds, nll, kl, bc, h, span_margs, \
                argmax_spans, trees, _ = model_output

        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

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
        total_bc += bc.sum().item()
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
            logging(
                'Test: [{0}/{1}]\t{e_log}\t'
                .format(
                    i, len(data_loader), e_log=str(model.logger)
                )
            )
        del images, captions

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    bc = total_bc / n_sent
    h = total_h / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}, BC: {:.4f}, H: {:.4f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, bc, h, corpus_f1 * 100, sent_f1 * 100
    )
    logging(info)
    return img_embs, cap_embs, ppl_elbo, sent_f1 * 100 

def validate_parser(opt, data_loader, model, vocab, logger):
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    nbatch = len(data_loader)
    
    # stats
    n_word, n_sent = 0, 0
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl, total_bc, total_h = 0., 0., 0., 0.

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (images, captions, lengths, ids, spans) in enumerate(data_loader):
        # make sure val logger is used
        del images
        model.logger = val_logger
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()

        # compute the embeddings
        bsize = captions.size(0) #cum_reward.shape[0]

        nll, kl, bc, h, _, argmax_spans, trees, lprobs = model.forward_parser(captions, lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_ll += nll.sum().item()
        total_kl += kl.sum().item()
        total_bc += bc.sum().item()
        total_h += h.sum().item()
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        # F1
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

        if i % model.log_step == 0:
            logger.info(
                'Test: [{0}/{1}]\t{e_log}\t'
                .format(
                    i, nbatch, e_log=str(model.logger)
                )
            )
        del captions, lengths, ids, spans
        #if i > 10: break

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    bc = total_bc / n_sent
    h = total_h / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}, BC: {:.4f}, H: {:.4f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, bc, h, corpus_f1 * 100, sent_f1 * 100
    )
    logger.info(info)
    return ppl_elbo 


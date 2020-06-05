import os, sys
import pickle, argparse
import numpy as np
from collections import defaultdict

import torch
from torch_struct import SentCFG
from torch_struct.networks import NeuralCFG, RoughCCFG

import data
import utils
from model import VGNSLCFGs
from utils import Vocabulary

def build_parse(spans, caption, vocab):
    tree = [(i, vocab.idx2word[int(word)]) for i, word in enumerate(caption)]
    tree = dict(tree)
    for l, r, A in spans:
        if l != r:
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span
    return tree[0] 

def eval_trees(model_path, vocab_name, split='test', gold='test_ground-truth.txt'):
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint = torch.load(model_path, map_location='cpu')
    opt = checkpoint['opt']

    # load vocabulary used by the model
    vocab_name = getattr(opt, "vocab_name", vocab_name)
    vocab = pickle.load(open(os.path.join(opt.data_path, vocab_name), 'rb'))
    opt.vocab_size = len(vocab)
    opt.vse_rl_alpha=0.0
    opt.vse_mt_alpha=0.0
    opt.vse_lm_alpha=1.0
    opt.vse_bc_alpha=0.5
    opt.vse_h_alpha=0.00

    """
    # construct model
    model = VGNSLCFGs(opt)
    model.logger = print

    # load model state
    model.set_state_dict(checkpoint['model'])
    model.train()
    """
    args = opt
    parser = RoughCCFG(args.vocab_size, args.nt_states, args.t_states, 
                       h_dim = args.h_dim,
                       w_dim = args.w_dim,
                       z_dim = args.z_dim,
                       s_dim = args.state_dim)
    parser_params = checkpoint['model'][VGNSLCFGs.NS_PARSER]
    parser.load_state_dict(parser_params)
    parser.cuda()
    parser.eval()

    batch_size = 5 
    prefix = getattr(opt, "prefix", "")
    print('Loading dataset', opt.data_path + prefix + split)
    data_loader = data.eval_data_iter(opt.data_path, prefix + split, vocab, batch_size=batch_size)

    # stats
    trees = list()
    n_word, n_sent = 0, 0
    per_label_f1 = defaultdict(list) 
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl, total_bc, total_h = 0., 0., 0., 0.

    for i, (captions, lengths, spans, labels, tags, ids) in enumerate(data_loader):
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()

        params, xx = parser(captions)
        dist = SentCFG(params, lengths=lengths)
        
        arg_spans = dist.argmax[-1]
        argmax_spans, _, _ = utils.extract_parses(arg_spans, lengths.tolist(), inc=0) 

        candidate_trees = list()
        bsize = captions.shape[0]
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

        for b in range(bsize):
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(l, r) for l, r in spans[b]] 
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
            
            word_tree = build_parse(argmax_spans[b], captions[b].tolist(), vocab) 
            candidate_trees.append(word_tree)

            for j, gold_span in enumerate(gold[:-1]):
                label = labels[b][j]
                per_label_f1.setdefault(label, [0., 0.]) 
                per_label_f1[label][0] += 1
                if gold_span in pred_set:
                    per_label_f1[label][1] += 1 

        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            tree = candidate_trees[j]
            appended_trees[ids[j] - min(ids)] = tree 
        for tree in appended_trees:
            print(tree)
        trees.extend(appended_trees)
        
        #if i == 50: break

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    print(info)

    print("\nPER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        
    return trees

def test_trees(model_path, split='test', gold='test_ground-truth.txt'):
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint = torch.load(model_path, map_location='cpu')
    opt = checkpoint['opt']

    #print(opt)
    #sys.exit(0)

    # load vocabulary used by the model
    vocab = pickle.load(open(os.path.join(opt.data_path, "coco.dict.pkl"), 'rb'))
    opt.vocab_size = len(vocab)
    opt.vse_rl_alpha=0.0
    opt.vse_mt_alpha=0.0
    opt.vse_lm_alpha=1.0
    opt.vse_bc_alpha=0.5
    opt.vse_h_alpha=0.00

    # construct model
    model = VGNSLCFGs(opt)

    # load model state
    print('Loading model', model_path)
    model.set_state_dict(checkpoint['model'])
    model.eval()

    print('Loading dataset', opt.data_path)
    data_loader = data.get_eval_iter(opt.data_path, split, vocab, 
        batch_size=opt.batch_size, shuffle=False)

    # stats
    trees = list()
    n_word, n_sent = 0, 0
    total_ll, total_kl = 0., 0.
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    for i, (_, captions, lengths, ids, spans) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = print
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()

        # compute the embeddings
        params, _ = model.parser(captions, use_mean=True)
        dist = SentCFG(params, lengths=lengths)
        
        arg_spans = dist.argmax[-1]
        argmax_spans, _, _ = utils.extract_parses(arg_spans, lengths.tolist(), inc=0) 

        candidate_trees = list()
        bsize = captions.shape[0]
        n_word += (lengths + 1).sum().item()
        n_sent += bsize

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
            
            word_tree = build_parse(argmax_spans[b], captions[b].tolist(), vocab) 
            candidate_trees.append(word_tree)

        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            tree = candidate_trees[j]
            appended_trees[ids[j] - min(ids)] = tree 
        for tree in appended_trees:
            print(tree)
        trees.extend(appended_trees)

        #if i > 10: break

    tp, fp, fn = corpus_f1  
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
    recon_ppl = np.exp(total_ll / n_word)
    ppl_elbo = np.exp((total_ll + total_kl) / n_word) 
    kl = total_kl / n_sent
    info = '\nReconPPL: {:.2f}, KL: {:.4f}, PPL (Upper Bound): {:.2f}\n' + \
           'Corpus F1: {:.2f}, Sentence F1: {:.2f}'
    info = info.format(
        recon_ppl, kl, ppl_elbo, corpus_f1 * 100, sent_f1 * 100
    )
    print(info)

    return trees


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=str, required=True,
                        help='model path to evaluate')
    parser.add_argument('--data_name', type=str, default='test',
                        help='model path to evaluate')
    parser.add_argument('--gold_name', type=str, default='test_ground-truth.txt',
                        help='model path to evaluate')
    parser.add_argument('--vocab_name', type=str, default='vocabulary',
                        help='model path to evaluate')
    args = parser.parse_args()
    #test_trees(args.candidate, split=args.data_name, gold=args.gold_name)
    eval_trees(args.candidate, args.vocab_name, split=args.data_name, gold=args.gold_name)


import os, re
import numpy as np
import json, pickle, argparse
from collections import defaultdict

import torch
from torch_struct import SentCFG

import data
import utils
from utils import Vocabulary
from module import CompoundCFG

def build_parse(spans, caption, vocab):
    tree = [(i, vocab.idx2word[int(word)]) for i, word in enumerate(caption)]
    tree = dict(tree)
    for l, r, A in spans:
        if l != r:
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span
    return tree[0] 

def make_model(best_model, args):
    model = CompoundCFG(
        args.vocab_size, args.nt_states, args.t_states, 
        h_dim = args.h_dim,
        w_dim = args.w_dim,
        z_dim = args.z_dim,
        s_dim = args.state_dim
    )
    best_model = best_model['parser']
    model.load_state_dict(best_model)
    return model

def eval_trees(args):
    checkpoint = torch.load(args.model, map_location='cpu')
    opt = checkpoint['opt']
    use_mean = True
    # load vocabulary used by the model
    data_path = args.data_path
    #data_path = getattr(opt, "data_path", args.data_path)
    vocab_name = getattr(opt, "vocab_name", args.vocab_name)
    vocab = pickle.load(open(os.path.join(data_path, vocab_name), 'rb'))
    checkpoint['word2idx'] = vocab.word2idx
    opt.vocab_size = len(vocab)
    
    parser = checkpoint['model']
    parser = make_model(parser, opt)
    parser.cuda()
    parser.eval()

    batch_size = 5 
    prefix = args.prefix
    print('Loading dataset', data_path + prefix + args.split)
    data_loader = data.eval_data_iter(data_path, prefix + args.split, vocab, batch_size=batch_size)

    # stats
    trees = list()
    n_word, n_sent = 0, 0
    per_label_f1 = defaultdict(list) 
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = [], [0., 0., 0.] 
    total_ll, total_kl, total_bc, total_h = 0., 0., 0., 0.

    pred_out = open(args.out_file, "w")

    for i, (captions, lengths, spans, labels, tags, ids) in enumerate(data_loader):
        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths
        if torch.cuda.is_available():
            lengths = lengths.cuda()
            captions = captions.cuda()

        params, kl = parser(captions, lengths, use_mean=use_mean)
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
            gold = [(l, r) for l, r in spans[b] if l != r] 
            gold_set = set(gold[:-1])

            ccaption = captions[b].tolist()[:max_len]
            sent = [vocab.idx2word[int(word)] for _, word in enumerate(ccaption)]
            iitem = (sent, gold, labels, pred) 
            json.dump(iitem, pred_out)
            pred_out.write("\n")

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
                label = re.split("=|-", label)[0]
                per_label_f1.setdefault(label, [0., 0.]) 
                per_label_f1[label][0] += 1

                lspan = gold_span[1] - gold_span[0] + 1
                by_length_f1.setdefault(lspan, [0., 0.])
                by_length_f1[lspan][0] += 1

                if gold_span in pred_set:
                    per_label_f1[label][1] += 1 
                    by_length_f1[lspan][1] += 1

        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            tree = candidate_trees[j]
            appended_trees[ids[j] - min(ids)] = tree 
        for tree in appended_trees:
            #print(tree)
            pass
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

    f1_ids=["CF1", "SF1", "NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]

    f1s = {"CF1": corpus_f1, "SF1": sent_f1} 

    print("\nPER-LABEL-F1 (label, acc)\n")
    for k, v in per_label_f1.items():
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        f1s[k] = v[1] / v[0]

    f1s = ['{:.2f}'.format(float(f1s[x]) * 100) for x in f1_ids] 
    print("\t".join(f1_ids))
    print("\t".join(f1s))

    acc = []

    print("\nPER-LENGTH-F1 (length, acc)\n")
    xx = sorted(list(by_length_f1.items()), key=lambda x: x[0])
    for k, v in xx:
        print("{}\t{:.4f} = {}/{}".format(k, v[1] / v[0], v[1], v[0]))
        if v[0] >= 5:
            acc.append((str(k), '{:.2f}'.format(v[1] / v[0])))
    k = [x for x, _ in acc]
    v = [x for _, x in acc]
    print("\t".join(k))
    print("\t".join(v))
        
    pred_out.close()
    return trees

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--split', type=str, default='test_gold')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--data_path', type=str, default='./mscoco')
    parser.add_argument('--vocab_name', type=str, default='coco.dict.pkl')
    parser.add_argument('--out_file', default='pred-parse.txt')
    parser.add_argument('--gold_out_file', default='gold-parse.txt')
    args = parser.parse_args()
    eval_trees(args)

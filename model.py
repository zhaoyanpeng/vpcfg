import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_struct import SentCFG

import utils
from module import CompoundCFG

class VGCPCFGs(object):
    NS_PARSER = 'parser'
    NS_OPTIMIZER = 'optimizer'

    def __init__(self, opt, vocab, logger):
        self.niter = 0
        self.vocab = vocab
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip
        self.vse_lm_alpha = opt.vse_lm_alpha
        
        self.parser = CompoundCFG(
            opt.vocab_size, opt.nt_states, opt.t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )

        self.all_params = [] 
        self.all_params += list(self.parser.parameters())
        self.optimizer = torch.optim.Adam(
            self.all_params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        ) 

        if torch.cuda.is_available():
            cudnn.benchmark = False 
            self.parser.cuda()
        logger.info(self.parser)

    def train(self):
        self.parser.train()

    def eval(self):
        self.parser.eval()

    def get_state_dict(self):
        state_dict = { 
            self.NS_PARSER: self.parser.state_dict(), 
            self.NS_OPTIMIZER: self.optimizer.state_dict(),
        } 
        return state_dict

    def set_state_dict(self, state_dict):
        self.parser.load_state_dict(state_dict[self.NS_PARSER])
        self.optimizer.load_state_dict(state_dict[self.NS_OPTIMIZER])

    def norms(self):
        p_norm = sum([p.norm() ** 2 for p in self.all_params]).item() ** 0.5
        g_norm = sum([p.grad.norm() ** 2 for p in self.all_params if p.grad is not None]).item() ** 0.5
        return p_norm, g_norm

    def forward_parser(self, captions, lengths):
        params, kl = self.parser(captions)
        dist = SentCFG(params, lengths=lengths)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = utils.extract_parses(the_spans, lengths.tolist(), inc=0) 

        ll = dist.partition
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return nll, kl, argmax_spans, trees, lprobs

    def forward(self, images, captions, lengths, ids=None, spans=None, epoch=None, *args):
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths
        if torch.cuda.is_available():
            captions = captions.cuda()
            lengths = lengths.cuda()
        bsize = captions.size(0) 

        nll, kl, argmax_spans, trees, lprobs = self.forward_parser(captions, lengths)

        ll_loss = nll.sum()
        kl_loss = kl.sum()
        
        loss = self.vse_lm_alpha * (ll_loss + kl_loss) / bsize

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.all_params, self.grad_clip)
        self.optimizer.step()
        
        self.logger.update('Loss', loss.item(), bsize)
        self.logger.update('KL-Loss', kl_loss.item() / bsize, bsize)
        self.logger.update('LL-Loss', ll_loss.item() / bsize, bsize)

        self.n_word += (lengths + 1).sum().item()
        self.n_sent += bsize

        for b in range(bsize):
            max_len = lengths[b].item() 
            pred = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]]
            pred_set = set(pred[:-1])
            gold = [(spans[b][i][0].item(), spans[b][i][1].item()) for i in range(max_len - 1)] 
            gold_set = set(gold[:-1])
            utils.update_stats(pred_set, [gold_set], self.all_stats) 

        info = ''
        if self.niter % self.log_step == 0:
            p_norm, g_norm = self.norms()
            all_f1 = utils.get_f1(self.all_stats)
            train_kl = self.logger.meters["KL-Loss"].sum 
            train_ll = self.logger.meters["LL-Loss"].sum 
            info = '|Pnorm|: {:.6f}, |Gnorm|: {:.2f}, ReconPPL: {:.2f}, KL: {:.2f}, ' + \
                   'PPLBound: {:.2f}, CorpusF1: {:.2f}, Speed: {:.2f} sents/sec'
            info = info.format(
                p_norm, g_norm, np.exp(train_ll / self.n_word), train_kl / self.n_sent,
                np.exp((train_ll + train_kl) / self.n_word), all_f1[0], 
                self.n_sent / (time.time() - self.s_time)
            )
            pred_action = utils.get_actions(trees[0])
            sent_s = [self.vocab.idx2word[wid] for wid in captions[0].cpu().tolist()]
            pred_t = utils.get_tree(pred_action, sent_s)
            gold_t = utils.span_to_tree(spans[0].tolist(), lengths[0].item()) 
            gold_action = utils.get_actions(gold_t) 
            gold_t = utils.get_tree(gold_action, sent_s)
            info += "\nPred T: {}\nGold T: {}".format(pred_t, gold_t)
        return info

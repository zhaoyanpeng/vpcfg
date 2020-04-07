import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from torch.nn.utils.clip_grad import clip_grad_norm_

import utils

from torch_struct import SentCFG
from torch_struct.networks import RoughCCFG


class VGNSLCFGs(object):
    NS_PARSER = 'parser'
    NS_OPTIMIZER = 'optimizer'

    def __init__(self, opt):
        self.parser = RoughCCFG(
            opt.vocab_size, opt.nt_states, opt.t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )
        self.NT = opt.nt_states
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip

        if torch.cuda.is_available():
            self.parser.cuda()
            cudnn.benchmark = False 

        self.vse_rl_alpha = opt.vse_rl_alpha
        self.vse_mt_alpha = opt.vse_mt_alpha
        self.vse_lm_alpha = opt.vse_lm_alpha
        self.vse_bc_alpha = opt.vse_bc_alpha
        self.vse_h_alpha = opt.vse_h_alpha

        self.all_params = [] 
        self.all_params += list(self.parser.parameters())

        self.optimizer = torch.optim.Adam(
            self.all_params, lr=opt.learning_rate, betas=(opt.beta1, opt.beta2)
        ) 
        self.niter = 0

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
        params, kl = self.parser(captions, use_mean=not self.parser.training)
        dist = SentCFG(params, lengths=lengths)

        rule_H = torch.tensor(0.0, device=captions.device)
        bc_coe = torch.tensor(0.0, device=captions.device)
        if self.vse_h_alpha > 0. or self.vse_bc_alpha > 0.:
            bsize = captions.size(0)
            rule_lprobs = params[1].view(bsize, self.NT, -1)
            rule_probs = rule_lprobs.exp()
            if self.vse_h_alpha > 0.:
                rule_H = -(rule_probs * rule_lprobs).sum(-1).mean(-1)
            if self.vse_bc_alpha > 0.:
                bc_coe = torch.matmul(rule_probs, rule_probs.transpose(-1, -2))
                I = torch.arange(self.NT, device=captions.device).long()
                bc_coe[:, I, I] = 0 
                bc_coe = bc_coe.sqrt().sum(-1).mean(-1)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = utils.extract_parses(the_spans, lengths.tolist(), inc=0) 

        #ll, _ = dist.partition
        #ll, _ = dist.inside
        ll, _ = dist.inside_bp
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl

        span_marg_matrix = None 
        return nll, kl, bc_coe, rule_H, span_marg_matrix, argmax_spans, trees, lprobs

    def train_parser(self, images, captions, lengths, ids=None, spans=None, epoch=None, *args):
        """ one training step given images and captions """
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        if torch.cuda.is_available():
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths).long()
            lengths = lengths.cuda()
            captions = captions.cuda()
        bsize = images.size(0) 

        nll, kl, bc, h, _, argmax_spans, trees, lprobs = self.forward_parser(captions, lengths)

        rl_loss = torch.tensor(0.0, device=nll.device) 
        mt_loss = torch.tensor(0.0, device=nll.device)        

        ll_loss = nll.sum()
        kl_loss = kl.sum()
        bc_loss = bc.sum()
        h_loss = h.sum()
        
        loss = self.vse_rl_alpha * rl_loss + \
               self.vse_mt_alpha * mt_loss + \
               self.vse_lm_alpha * (ll_loss + kl_loss) + \
               self.vse_bc_alpha * bc_loss + \
               self.vse_h_alpha * h_loss
        loss = loss / bsize  

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.all_params, self.grad_clip)
        self.optimizer.step()
        
        # log things
        self.logger.update('Loss', loss.item(), bsize)
        self.logger.update('H-Loss', h_loss.item() / bsize, bsize)
        self.logger.update('BC-Loss', bc_loss.item() / bsize, bsize)
        self.logger.update('MT-Loss', mt_loss.item() / bsize, bsize)
        self.logger.update('RL-Loss', rl_loss.item() / bsize, bsize)
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


import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch_struct import SentCFG

import utils
from module import CompoundCFG, ContrastiveLoss, ImageEncoder, TextEncoder

class VGCPCFGs(object):
    NS_PARSER = 'parser'
    NS_TXT_ENCODER = 'txt_enc'
    NS_IMG_ENCODER = 'img_enc' 
    NS_OPTIMIZER = 'optimizer'

    def __init__(self, opt, vocab, logger):
        self.niter = 0
        self.vocab = vocab
        self.logger = logger
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip

        self.vse_mt_alpha = opt.vse_mt_alpha
        self.vse_lm_alpha = opt.vse_lm_alpha

        self.loss_criterion = ContrastiveLoss(margin=opt.margin)

        self.parser = CompoundCFG(
            opt.vocab_size, opt.nt_states, opt.t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )

        word_emb = torch.nn.Embedding(len(vocab), opt.word_dim)
        torch.nn.init.xavier_uniform_(word_emb.weight)

        self.all_params = [] 
        self.img_enc = ImageEncoder(opt)
        self.txt_enc = TextEncoder(opt, word_emb)
        self.all_params += list(self.txt_enc.parameters())
        self.all_params += list(self.parser.parameters())
        self.all_params += list(self.img_enc.parameters())
        self.optimizer = torch.optim.Adam(
            self.all_params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        ) 

        if torch.cuda.is_available():
            cudnn.benchmark = False 
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.parser.cuda()
        self.logger.info(self.parser)

    def train(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.parser.train()

    def eval(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.parser.eval()

    def get_state_dict(self):
        state_dict = { 
            self.NS_PARSER: self.parser.state_dict(), 
            self.NS_IMG_ENCODER: self.img_enc.state_dict(), 
            self.NS_TXT_ENCODER: self.txt_enc.state_dict(), 
            self.NS_OPTIMIZER: self.optimizer.state_dict(),
        } 
        return state_dict

    def set_state_dict(self, state_dict):
        self.parser.load_state_dict(state_dict[self.NS_PARSER])
        self.img_enc.load_state_dict(state_dict[self.NS_IMG_ENCODER]) 
        self.txt_enc.load_state_dict(state_dict[self.NS_TXT_ENCODER]) 
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

        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return nll, kl, span_margs, argmax_spans, trees, lprobs

    def forward_encoder(self, images, captions, lengths, spans, require_grad=True):
        if torch.cuda.is_available():
            images = images.cuda()
            lengths = lengths.cuda()
            captions = captions.cuda()
        with torch.set_grad_enabled(require_grad):
            img_emb = self.img_enc(images)
            parser_outs = self.forward_parser(captions, lengths) 
            txt_outputs = self.txt_enc(captions, lengths, parser_outs[-3])
        return (img_emb, txt_outputs) + parser_outs

    def forward_loss(self, base_img_emb, cap_span_features, lengths, span_bounds, span_margs):
        b = base_img_emb.size(0)
        N = lengths.max(0)[0]
        nstep = int(N * (N - 1) / 2)
        mstep = (lengths * (lengths - 1) / 2).int()
        # focus on only short spans
        nstep = int(mstep.float().mean().item() / 2)

        matching_loss_matrix = torch.zeros(
            b, nstep, device=base_img_emb.device
        )
        for k in range(nstep):
            img_emb = base_img_emb
            cap_emb = cap_span_features[:, k] 
            
            cap_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)

            cap_emb = utils.l2norm(cap_emb) 
            loss = self.loss_criterion(img_emb, cap_emb)
            matching_loss_matrix[:, k] = loss
        span_margs = span_margs.sum(-1)
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix 
        expected_loss = expected_loss.sum(-1)
        return expected_loss

    def forward(self, images, captions, lengths, ids=None, spans=None, epoch=None, *args):
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        img_emb, cap_span_features, nll, kl, span_margs, argmax_spans, trees, lprobs = \
            self.forward_encoder(
                images, captions, lengths, spans
            )
        matching_loss = self.forward_loss(
            img_emb, cap_span_features, lengths, argmax_spans, span_margs
        )

        bsize = captions.size(0) 

        rl_loss = torch.tensor(0.0, device=nll.device) 
        mt_loss = matching_loss.sum()

        kl.clamp_(max=20) # avoid kl explosion
        if self.vse_lm_alpha <=0.:
            kl = torch.zeros_like(kl) 
            nll = torch.zeros_like(nll) 

        ll_loss = nll.sum()
        kl_loss = kl.sum()
        
        loss = (self.vse_mt_alpha * mt_loss + self.vse_lm_alpha * (ll_loss + kl_loss)) / bsize 

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.all_params, self.grad_clip)
        self.optimizer.step()
        
        self.logger.update('Loss', loss.item(), bsize)
        self.logger.update('MT-Loss', mt_loss.item() / bsize, bsize)
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
        if epoch > 0: #
            del img_emb, cap_span_features, left_span_features, right_span_features, \
                word_embs, tree_indices, probs, span_bounds, nll, kl, span_margs, \
                argmax_spans, trees, lprobs, matching_loss 
        return info

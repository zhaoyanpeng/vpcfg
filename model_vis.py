import os, sys, time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_

import utils
from data import PartiallyFixedEmbedding
from module import ContrastiveLoss, ImageEncoder, TextEncoder

from torch_struct import SentCFG
from torch_struct.networks import RoughCCFG, CompoundCFG
from torch_struct.networks import ACompPCFG as YoonPCFG


class VGCPCFGs(object):
    NS_PARSER = 'parser'
    NS_TXT_ENCODER = 'txt_enc'
    NS_IMG_ENCODER = 'img_enc' 
    NS_OPTIMIZER = 'optimizer'
    def __init__(self, opt, vocab, log):
        self.vocab = vocab
        self.NT = opt.nt_states

        self.niter = 0
        self.log = log
        self.log_step = opt.log_step
        self.grad_clip = opt.grad_clip

        self.vse_rl_alpha = opt.vse_rl_alpha
        self.vse_mt_alpha = opt.vse_mt_alpha
        self.vse_lm_alpha = opt.vse_lm_alpha
        self.vse_bc_alpha = opt.vse_bc_alpha
        self.vse_h_alpha = opt.vse_h_alpha

        self.loss_criterion = ContrastiveLoss(margin=opt.margin)

        if opt.parser_type == '1st':
            parser = RoughCCFG  
        elif opt.parser_type == '2nd':
            parser = CompoundCFG 
        elif opt.parser_type == '3rd':
            parser = YoonPCFG 
        else:
            raise NameError("Invalid parser type: {}".format(opt.parser_type)) 
        self.parser = parser(
            opt.vocab_size, opt.nt_states, opt.t_states, 
            h_dim = opt.h_dim,
            w_dim = opt.w_dim,
            z_dim = opt.z_dim,
            s_dim = opt.state_dim
        )

        w2vec_file = opt.data_path + opt.w2vec_file
        if os.path.isfile(w2vec_file):
            word_emb = PartiallyFixedEmbedding(
                self.vocab, w2vec_file, opt.word_dim
            )
        else:
            word_emb = torch.nn.Embedding(len(vocab), opt.word_dim)
            torch.nn.init.xavier_uniform_(word_emb.weight)

        self.all_params = [] 
        self.img_enc = ImageEncoder(opt)
        if opt.share_w2vec:
            self.parser.enc_emb = word_emb 
            self.txt_enc = TextEncoder(opt, None)
            self.all_params += list(self.txt_enc.parameters())
            self.txt_enc.set_enc_emb(word_emb) # word_emb optimized once
        else:
            self.txt_enc = TextEncoder(opt, word_emb)
            self.all_params += list(self.txt_enc.parameters())
        self.all_params += list(self.parser.parameters())
        self.all_params += list(self.img_enc.parameters())
        self.optimizer = torch.optim.Adam(
            self.all_params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        ) 

        if torch.cuda.is_available():
            self.parser.cuda()
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = False 

        self.log.info(self.parser)
        p_emb, t_emb = None, None
        for k, v in self.parser.named_parameters():
            if "enc_emb" in k:
                p_emb = v
                self.log.info("P: {} {}".format(k, v.size()))
        for k, v in self.txt_enc.named_parameters():
            if "enc_emb" in k:
                t_emb = v
                self.log.info("T: {} {}".format(k, v.size()))
        if opt.share_w2vec:
            #assert p_emb == t_emb
            pass

    def train(self):
        self.parser.train()
        self.img_enc.train()
        self.txt_enc.train()

    def eval(self):
        self.parser.eval()
        self.img_enc.eval()
        self.txt_enc.eval()

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
        #params, kl = self.parser(captions, use_mean=not self.parser.training)
        params, kl = self.parser(captions)
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
        #ll, _ = dist.inside_bp
        #ll    = dist.partition
        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl

        return nll, kl, bc_coe, rule_H, span_margs, argmax_spans, trees, lprobs

    def forward_encoder(self, images, captions, lengths, spans, require_grad=True):
        if torch.cuda.is_available():
            #spans = spans.cuda()
            images = images.cuda()
            lengths = lengths.cuda()
            captions = captions.cuda()
        with torch.set_grad_enabled(require_grad):
            img_emb = self.img_enc(images)
            parser_outs = self.forward_parser(captions, lengths) 
            txt_outputs = self.txt_enc(captions, lengths, parser_outs[-3])
        return (img_emb, ) + txt_outputs + parser_outs

    def forward_loss(self, 
        base_img_emb, cap_span_features, left_span_features, right_span_features, 
        word_embs, lengths, span_bounds, span_margs):

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
        #expected_loss = expected_loss.sum(-1) / b 
        return None, expected_loss

    def forward(self, images, captions, lengths, ids=None, spans=None, epoch=None, *args):
        self.niter += 1
        self.logger.update('Eit', self.niter)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        lengths = torch.tensor(lengths).long() if isinstance(lengths, list) else lengths

        #print(lengths)

        img_emb, cap_span_features, left_span_features, right_span_features, \
            word_embs, tree_indices, probs, span_bounds, nll, kl, bc, h, span_margs, \
                argmax_spans, trees, lprobs = self.forward_encoder(
            images, captions, lengths, spans
        )

        cum_reward, matching_loss = self.forward_loss(
            img_emb, cap_span_features, left_span_features, right_span_features, 
            word_embs, lengths, argmax_spans, span_margs
        )

        bsize = images.size(0) 

        rl_loss = torch.tensor(0.0, device=nll.device) 
        mt_loss = matching_loss.sum() #torch.tensor(0.0, device=nll.device)        

        kl.clamp_(max=20) # avoid kl explosion
        if self.vse_lm_alpha <=0.:
            kl = torch.zeros_like(kl) 
            nll = torch.zeros_like(nll) 

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
        #self.logger.update('H-Loss', h_loss.item() / bsize, bsize)
        #self.logger.update('BC-Loss', bc_loss.item() / bsize, bsize)
        self.logger.update('MT-Loss', mt_loss.item() / bsize, bsize)
        #self.logger.update('RL-Loss', rl_loss.item() / bsize, bsize)
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
                word_embs, tree_indices, probs, span_bounds, nll, kl, bc, h, span_margs, \
                argmax_spans, trees, lprobs, cum_reward, matching_loss 
        return info


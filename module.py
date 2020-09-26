import torch
from torch import nn
import torch.nn.functional as F

from utils import l2norm, cosine_sim

class ResLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class CompoundCFG(torch.nn.Module):
    def __init__(self, V, NT, T, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256): 
        super(CompoundCFG, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.root_emb = nn.Parameter(torch.randn(1, s_dim))
        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))

        self.rule_mlp = nn.Linear(s_dim + z_dim, self.NT_T ** 2)
        self.root_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        self.term_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        if z_dim > 0:
            self.enc_emb = nn.Embedding(V, w_dim)
            self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                bidirectional=True, num_layers=1, batch_first=True)
            self.enc_out = nn.Linear(h_dim * 2, z_dim * 2)
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x):
        x_embbed = self.enc_emb(x)
        h, _ = self.enc_rnn(x_embbed)
        out = self.enc_out(h.max(1)[0])
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar
    
    def forward(self, x, *args, use_mean=False, **kwargs):
        b, n = x.shape[:2] 
        if self.z_dim > 0:
            mean, lvar = self.enc(x)
            kl = self.kl(mean, lvar).sum(1)
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                    b, n, self.T, self.z_dim
                )
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, img, txt):
        scores = self.sim(img, txt)
        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img

class ImageEncoder(torch.nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.no_imgnorm = opt.no_imgnorm
        self.fc = torch.nn.Linear(opt.img_dim, opt.sem_dim)
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, images):
        # why: assuming that the precomputed features are already l2-normalized
        features = self.fc(images.float())
        if not self.no_imgnorm:
            features = l2norm(features)
        return features

class TextEncoder(torch.nn.Module):
    def __init__(self, opt, enc_emb=None):
        super(TextEncoder, self).__init__()
        self.NT = opt.nt_states
        self.sem_dim = opt.sem_dim
        self.syn_dim = opt.syn_dim
        self.enc_rnn = torch.nn.LSTM(opt.word_dim, opt.lstm_dim, 
            bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = torch.nn.Linear(
            opt.lstm_dim * 2, self.NT * self.sem_dim
        )
        self._initialize()
        self.enc_emb = enc_emb # avoid double initialization 

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def set_enc_emb(self, enc_emb):
        self.enc_emb = enc_emb

    def _forward_srnn(self, x_emb, lengths, spans=None):
        """ lstm over every span, a.k.a. segmental rnn 
        """
        b, N, dim = x_emb.size()
        assert N == lengths.max()
        word_mask = torch.arange(
            0, N, device=x_emb.device
        ).unsqueeze(0).expand(b, N).long() 
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        feats = torch.zeros(
            b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device
        )
        beg_idx = 0 
        for k in range(1, N):
            inc = torch.arange(N - k, device=x_emb.device).view(N - k, 1)#.expand(N - k, k + 1)
            idx = torch.arange(k + 1, device=x_emb.device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(b, -1, dim) 

            feat = torch.gather(word_vect, 1, idx)
            feat = feat.view(b, N - k, k + 1, dim)
            feat = feat.view(-1, k + 1, dim) 
            feat = self.enc_out(self.enc_rnn(feat)[0])
            feat = feat.view(b, N - k, k + 1, self.NT, self.sem_dim)
            feat = l2norm(feat.sum(2))
            end_idx = beg_idx + N - k 
            feats[:, beg_idx : end_idx] = feat 
            beg_idx = end_idx
        return feats 
    
    def forward(self, x, lengths, spans):  
        word_emb = self.enc_emb(x)
        return self._forward_srnn(word_emb, lengths)

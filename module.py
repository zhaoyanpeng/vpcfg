import torch

from utils import l2norm, cosine_sim


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

        self.encode_span = opt.encode_span
        
        if "lstm" in self.encode_span:
            self.enc_rnn = torch.nn.LSTM(opt.word_dim, opt.lstm_dim, 
                bidirectional=True, num_layers=1, batch_first=True)
            self.enc_out = torch.nn.Linear(
                opt.lstm_dim * 2, self.NT * self.sem_dim
            )
        else:
            self.enc_rnn = lambda x: (x, None) # dummy lstm 
            self.enc_out = torch.nn.Linear(opt.word_dim, self.NT * self.sem_dim)
        self._initialize()
        self.enc_emb = enc_emb # avoid double initialization 

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def set_enc_emb(self, enc_emb):
        self.enc_emb = enc_emb

    def _forward_been(self, x_emb, lengths, spans=None):
        """ Use left and right boundary features of a span 
        """
        x_emb = self.enc_out(self.enc_rnn(x_emb)[0])

        b, N, dim = x_emb.size()
        assert N == lengths.max()
        word_mask = torch.arange(
            0, N, device=x_emb.device
        ).unsqueeze(0).expand(b, N).long() 
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        word_vect = word_vect.view(b, N, self.NT, self.sem_dim)
        
        feats = torch.zeros(
            b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device
        )
        beg_idx = 0 
        for k in range(1, N):
            l, r = torch.arange(N - k), torch.arange(k, N)
            lfeat, rfeat = word_vect[:, l], word_vect[:, r]
            end_idx = beg_idx + N - k 
            new_feats = l2norm(lfeat + rfeat)
            feats[:, beg_idx : end_idx] = new_feats 
            beg_idx = end_idx
        return feats, None, None, None, None, None, None 

    def _forward_mean(self, x_emb, lengths, spans=None):
        """ Use the average word features of a span 
        """
        x_emb = self.enc_out(self.enc_rnn(x_emb)[0])

        b, N, dim = x_emb.size()
        assert N == lengths.max()
        word_mask = torch.arange(
            0, N, device=x_emb.device
        ).unsqueeze(0).expand(b, N).long() 
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        word_vect = word_vect.view(b, N, self.NT, self.sem_dim)
        
        feats = torch.zeros(
            b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device
        )
        beg_idx = 0 
        for k in range(1, N):
            inc = torch.arange(N - k, device=x_emb.device).view(N - k, 1)#.expand(N - k, k + 1)
            idx = torch.arange(k + 1, device=x_emb.device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                b, -1, self.NT, self.sem_dim
            ) 
            feat = torch.gather(word_vect, 1, idx)
            feat = feat.view(b, N - k, k + 1, self.NT, self.sem_dim)
            feat = l2norm(feat.sum(2))
            end_idx = beg_idx + N - k 
            feats[:, beg_idx : end_idx] = feat 
            beg_idx = end_idx
        return feats, None, None, None, None, None, None 

    def _forward_srnn(self, x_emb, lengths, spans=None):
        """ lstm over every span, a.k.a. segmental rnn 
        """
        #x_emb = self.enc_out(self.enc_rnn(x_emb)[0])

        b, N, dim = x_emb.size()
        assert N == lengths.max()
        word_mask = torch.arange(
            0, N, device=x_emb.device
        ).unsqueeze(0).expand(b, N).long() 
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        #word_vect = word_vect.view(b, N, self.NT, self.sem_dim)
        
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
        return feats, None, None, None, None, None, None 
    
    def forward(self, x, lengths, spans):  
        """ (a) lstm_been: lstm encoder + span boundary features
            (b) lstm_mean/mean: lstm/word encoder + average word features 
            (c) lstm_srnn: lstm (span) encoder + average word features
        """
        word_emb = self.enc_emb(x)
        if "mean" in self.encode_span: # all words in a span
            return self._forward_mean(word_emb, lengths)
        elif "srnn" in self.encode_span: # span/segmental rnn model 
            return self._forward_srnn(word_emb, lengths)
        elif "been" in self.encode_span: # leading and ending words
            return self._forward_been(word_emb, lengths)


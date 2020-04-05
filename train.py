import os, sys
import time, argparse, logging
import pickle, shutil
import numpy as np
import torch

import data
from model import VGNSLCFGs
from utils import Vocabulary, save_checkpoint, adjust_learning_rate
from evaluation import AverageMeter, LogCollector, validate_parser 


def train(opt, train_loader, model, epoch, val_loader, vocab):
    # average meters to record the training statistics
    train_logger = LogCollector()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    nbatch = len(train_loader)
    # switch to train mode
    end = time.time()
    model.n_word = 0
    model.n_sent = 0
    model.s_time = end
    model.all_stats = [[0., 0., 0.]]
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger
        # Update the model
        info = model.train_parser(*train_data, epoch=epoch)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log info
        if model.niter % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] {e_log} {info}'
                .format(
                    epoch, i, nbatch, e_log=str(model.logger), info=info
                )
            )
        #break
        # validate at every val_step
        if model.niter % opt.val_step == 0:
            validate_parser(opt, val_loader, model, vocab, logger)

def debug(opt, model):
    data_iter = data.get_eval_iter(opt.data_path, "toy", vocab, 
        batch_size=opt.batch_size, shuffle=False)
    for images, captions, lengths, ids, spans in data_iter:
        print(images.size(), ids, lengths)
        print(captions)
        print(spans)
        print('\n')
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            spans = spans.cuda()
            images = images.cuda()
            lengths = lengths.cuda()
            captions = captions.cuda()
        #xx = model.forward_parser(captions, lengths) 
        #xx = model.txt_enc(captions, lengths, spans)
        train_logger = LogCollector()
        end = time.time()

        model.n_word = 0
        model.n_sent = 0
        model.s_time = end
        model.all_stats = [[0., 0., 0.]]
        model.logger = train_logger
        model.train()
        info = model.train_parser(images, captions, lengths, spans=spans, epoch=0)
        print(info)
        break
    import sys
    sys.exit(0)

if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()

    # Parser: Generative model parameters
    parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
    parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
    parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
    parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
    # Parser: Inference network parameters
    parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
    parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
    parser.add_argument('--gpu', default=-1, type=int, help='which gpu to use')

    # 
    parser.add_argument('--seed', default=3435, type=int, help='random seed')
    parser.add_argument('--model_init', default=None, type=str, help='random seed')

    parser.add_argument('--data_path', default='../data/mscoco',
                        help='path to datasets')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=35, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--word_dim', default=512, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--lstm_dim', default=512, type=int,
                        help='dimensionality of the lstm hidden embedding')
    parser.add_argument('--embed_size', default=512, type=int,
                        help='dimensionality of the joint embedding')
    parser.add_argument('--grad_clip', default=3., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='number of epochs to update the learning rate')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--log_step', default=10, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=500, type=int,
                        help='number of steps to run validation')
    parser.add_argument('--logger_name', default='../output/',
                        help='path to save the model and log')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='dimensionality of the image embedding')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--scoring_hidden_dim', type=int, default=128,
                        help='score hidden dim')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')

    parser.add_argument('--init_embeddings', type=int, default=0)
    parser.add_argument('--init_embeddings_type', choices=['override', 'partial', 'partial-fixed'], default='override')
    parser.add_argument('--init_embeddings_key', choices=['glove', 'fasttext'], default='override')
    parser.add_argument('--init_embeddings_partial_dim', type=int, default=0)

    parser.add_argument('--syntax_score', default='conv', choices=['conv', 'dynamic'])
    parser.add_argument('--syntax_dim', type=int, default=300)

    # For syntax_score == 'conv'
    parser.add_argument('--syntax_score_hidden', type=int, default=128)
    parser.add_argument('--syntax_score_kernel', type=int, default=5)
    parser.add_argument('--syntax_dropout', type=float, default=0.1)

    parser.add_argument('--syntax_tied_with_semantics', type=int, default=1)
    parser.add_argument('--syntax_embedding_norm_each_time', type=int, default=1)
    parser.add_argument('--semantics_embedding_norm_each_time', type=int, default=1)
    parser.add_argument('--semantics_rep', type=str, default='lstm')

    parser.add_argument('--vse_rl_alpha', type=float, default=1.0)
    parser.add_argument('--vse_mt_alpha', type=float, default=1.0)
    parser.add_argument('--vse_lm_alpha', type=float, default=1.0)
    parser.add_argument('--vse_bc_alpha', type=float, default=1.0)
    parser.add_argument('--vse_h_alpha', type=float, default=1.0)

    parser.add_argument('--lambda_hi', type=float, default=0,
                        help='penalization for head-initial inductive bias')
    opt = parser.parse_args()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    else:
        print('Creating {}'.format(opt.logger_name))
        os.mkdir(opt.logger_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info('cuda:{}@{}'.format(opt.gpu, os.uname().nodename))
    logger.info(opt)

    # load predefined vocabulary and pretrained word embeddings if applicable
    vocab = pickle.load(open(os.path.join(opt.data_path, "coco.dict.pkl"), 'rb'))
    opt.vocab_size = len(vocab)

    # Load data loaders
    if opt.batch_size <= 5:
        train_loader, val_loader = data.get_train_iters(
            opt.data_path, vocab, opt.batch_size, opt.workers
        )
    else:
        train_loader = data.get_eval_iter(
            opt.data_path, "train", vocab, opt.batch_size, 
            nworker=opt.workers, shuffle=False, sampler=True 
        )
        val_loader = data.get_eval_iter(
            opt.data_path, "val", vocab, int(opt.batch_size / 2), 
            nworker=opt.workers, shuffle=False, sampler=None 
        )

    # construct the model
    model = VGNSLCFGs(opt)
    model.vocab = vocab
    if opt.model_init:
        logger.info("override parser's params.")
        checkpoint = torch.load(opt.model_init, map_location='cpu')
        parser_params = checkpoint['model'][VGNSLCFGs.NS_PARSER]
        model.parser.load_state_dict(parser_params)

    #debug(opt, model)

    save_checkpoint({
        'epoch': -1,
        'model': model.get_state_dict(),
        'best_rsum': -1,
        'opt': opt,
        'Eiters': -1,
    }, False, -1, prefix=opt.logger_name + '/')

    best_rsum = 0
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, vocab)
        # evaluate on validation set using VSE metrics
        rsum = validate_parser(opt, val_loader, model, vocab, logger)
        #break
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.get_state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.niter,
        }, is_best, epoch, prefix=opt.logger_name + '/')

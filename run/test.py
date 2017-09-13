#Embedded file name: /home/dengcai/code/run/test.py
from __future__ import division
import sys
sys.path.append('..')
import time, os, cPickle
import dynet as dy
import models
from lib import Vocab, DataLoader, RawDataLoader
from config import Configurable

def test(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file, notag = False):
    if not test_file.endswith('.conll'):
        raw_test(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file, notag)
        return
    data_loader = DataLoader(test_file, num_buckets_test, vocab)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    for words, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size, shuffle=False):
        dy.renew_cg()
        if notag:
            outputs = parser.run(words, isTrain=False)
        else:
            outputs = parser.run(words, tags, isTrain=False)
        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1

    arcs, rels = [], []
    for result in results:
        for x, y in zip(result[0], result[1]):
            arcs.append(x)
            rels.append(y)
    idx = 0
    with open(test_file) as f:
        with open(output_file, 'w') as fo:
            for line in f.readlines():
                info = line.strip().split()
                if info:
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[6] = str(arcs[idx])
                    info[7] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')

    os.system('perl eval.pl -q -b -g %s -s %s -o tmp' % (test_file, output_file))
    os.system('tail -n 3 tmp > score_tmp')
    LAS, UAS = [float(line.strip().split()[-2]) for line in open('score_tmp').readlines()[:2]]
    print 'LAS %.2f, UAS %.2f'%(LAS, UAS)
    os.system('rm tmp score_tmp')
    return LAS, UAS

def raw_test(parser, vocab, num_buckets_test, test_batch_size, test_file, output_file, notag = True):
    data_loader = RawDataLoader(test_file, num_buckets_test, vocab)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    print 'Start'
    for words, tags in data_loader.get_batches(batch_size=test_batch_size, shuffle=False):
        dy.renew_cg()
        if notag:
            outputs = parser.run(words, isTrain=False)
        else:
            outputs = parser.run(words, tags, isTrain=False)
        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    print 'Finishing'
    arcs, rels = [], []
    for result in results:
        for x, y in zip(result[0], result[1]):
            arcs.append(x)
            rels.append(y)
    idx = 0
    word_idx = 1
    output_info = ['_'] *10
    with open(test_file) as f:
        with open(output_file, 'w') as fo:
            for line in f.readlines():
                info = line.strip().split()
                if info:
                    assert len(info) == 2, 'Illegal line: %s' % line
                    output_info[0] = str(word_idx)
                    output_info[1], output_info[3] = info[0], info[1]
                    output_info[6] = str(arcs[idx])
                    output_info[7] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(output_info) + '\n')
                    idx += 1
                    word_idx +=1
                else:
                    word_idx = 1
                    fo.write('\n')
                    
import argparse
#python test.py --dynet-gpu --config_file ../ckpt/default/config.cfg --model NotagParser --output_file result2 --notag True --test_file ../../sancl_data/norm_data/unlabeled_emails_2
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--output_file', default='here')
    argparser.add_argument('--notag', type = bool, default = False)
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, for_test = True)
    Parser = getattr(models, args.model)
    vocab = cPickle.load(open(config.load_vocab_path))
    if args.model == 'BaseParser':
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)
    elif args.model == 'SentParser':
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.choice_size, randn_init = True)
        parser.set_trainable_flags(True, True, True, True, True)
    elif args.model == 'NotagParser':
        parser = Parser(vocab, config.word_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)
    elif args.model == 'NotagParser_auxemb':
        parser = Parser(vocab, config.word_dims, config.aux_word_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)
    elif args.model == 'NotagParser_auxfeat':
        parser = Parser(vocab, config.word_dims, config.aux_word_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)     
    parser.load(config.load_model_path)
    test(parser, vocab, config.num_buckets_test, config.test_batch_size, config.test_file, args.output_file, args.notag)

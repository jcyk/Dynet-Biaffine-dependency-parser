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

    arcs, rels, probs = [], [], []
    for result in results:
        for x, y, z in zip(result[0], result[1], result[2]):
            arcs.append(x)
            rels.append(y)
            probs.append(z)
    idx = 0
    with open(test_file) as f:
        with open(output_file, 'w') as fo:
            for line in f.readlines():
                info = line.strip().split()
                if info:
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[5] = str(probs[idx])
                    info[6] = str(arcs[idx])
                    info[7] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')

    os.system('perl eval.pl -q -b -g %s -s %s -o %s' % (test_file, output_file, output_file+"tmp"))
    os.system('tail -n 3 %s > %s'%(output_file+"tmp",output_file+"score_tmp"))
    LAS, UAS = [float(line.strip().split()[-2]) for line in open(output_file+"score_tmp").readlines()[:2]]
    print 'LAS %.2f, UAS %.2f'%(LAS, UAS)
    os.system('rm %s %s'%(output_file+"tmp",output_file+"score_tmp"))
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
    arcs, rels, probs = [], [], []
    for result in results:
        for x, y, z in zip(result[0], result[1], result[2]):
            arcs.append(x)
            rels.append(y)
            probs.append(z)
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
                    output_info[5] = str(probs[idx])
                    output_info[6] = str(arcs[idx])
                    output_info[7] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(output_info) + '\n')
                    idx += 1
                    word_idx +=1
                else:
                    word_idx = 1
                    fo.write('\n')
                    
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    argparser.add_argument('--output_file', default='here')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, for_test = True)
    Parser = getattr(models, config.model)
    if config.model == "BaseParserMulti":
        from baseline_train_multi import MixedVocab
        vocab0 = cPickle.load(open(config.load_vocab_path+"0"))
        vocab1 = cPickle.load(open(config.load_vocab_path+"1"))
        vocab = MixedVocab([vocab0, vocab1])
    else:
        vocab = cPickle.load(open(config.load_vocab_path))
    _notag = False
    if config.model == 'BaseParser':
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)
    elif config.model == 'LossParser':
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.choice_size, randn_init = True)    
    elif config.model == 'NotagParser':
        _notag = True
        parser = Parser(vocab, config.word_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, randn_init = True)
    elif config.model == 'DistilltagParser':
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.choice_size, randn_init = True)    
    elif config.model == "BaseParserMulti":
        parser = Parser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp)
    parser.load(config.load_model_path)
    test(parser, vocab, config.num_buckets_test, config.test_batch_size, config.test_file, args.output_file, _notag)

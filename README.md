# Dynet-Biaffine-dependency-parser
This repository implements the parser described in the paper [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

I tried my best to **exactly copy every detail** in the [tensorflow code](https://github.com/tdozat/Parser) (which is the original code from the paper's authors), including the weight initialization, choice of activation fucntion (and dropout), data preprocess, batch operation, MST algorithm, and, of course, hyperparameters.

*For one who is interested, please tell me if you can find a difference.*

On some version of WSJ data. This code only acheived UAS 95.56%, though the original code acheived UAS 95.59%.

## Usage (by examples)

### Train
```
  cd run
  python train.py --config_file ../configs/default.cfg --save_dir ../ckpt/default
```

### Test
```
  cd run
  python test.py --config_file ../configs/default.cfg --load_dir ../ckpt/default --output_file here
```

All configuration options (see in `run/config.py`) can be specified on the command line, but it's much easier to instead store them in a configuration file like `configs/default.cfg`.

### Known Issues
Never stop? try [#6](https://github.com/jcyk/Dynet-Biaffine-dependency-parser/issues/6)

LAS is low? try [#2](https://github.com/jcyk/Dynet-Biaffine-dependency-parser/issues/2)


### Contact
Deng Cai, thisisjcykcd (AT) gmail.com

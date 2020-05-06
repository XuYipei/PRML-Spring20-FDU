
import argparse

from handout import *


if __name__ == '__main__':
    '''
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', choices=['rnn', 'lstm', 'gru'], default='rnn')
    parser.add_argument('--adv', choices=['rnn', 'irnn'], default='rnn')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train', choices=['normal', 'extreme'], default='normal')
    parser.add_argument('--evaluate', choices=['normal', 'extreme'], default='normal')
    arg = parser.parse_args()
    pt_main(choice=arg.pt, epoch=arg.epoch, batch_size=arg.batch_size, train_set=arg.train, evaluate_set=arg.evaluate)
    pt_adv_main(choice=arg.adv, epoch=arg.epoch, batch_size=arg.batch_size, train_set=arg.train, evaluate_set=arg.evaluate)
    
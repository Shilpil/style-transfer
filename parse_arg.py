# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:15:47 2019

@author: Shilpa
"""
from argparse import ArgumentParser
import train

CONTENT_WEIGHT = 5
TV_WEIGHT = 1e-3
INITIAL_LR = 1e1
MAX_ITER = 1000
STYLE_WEIGHTS = [10000, 10000, 10000, 10000, 10000]
PRINT_ITERATIONS = None
IMG_SIZE = 512
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--style',
            dest='style', help='style image',
            metavar='STYLE', required=True)
    parser.add_argument('--vgg_weights',
            dest='h5_file', help='h5 file containing VGG19 weights without the top layer',
            metavar='H5_FILE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
            dest='max_iter', help='iterations (default %(default)s)',
            metavar='MAX_ITER', default=MAX_ITER)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='checkpoint output frequency',
            metavar='PRINT_ITERATIONS', default=PRINT_ITERATIONS)
    parser.add_argument('--img_size', type=int,
            dest='img_size', help='output width',
            metavar='IMG_SIZE', default=IMG_SIZE)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight',
            help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--style-weights', type=float,
            dest='style_weights', help='style weights for the 5 layers',
            nargs='+', metavar='STYLE_WEIGHTS',default=STYLE_WEIGHTS)
    parser.add_argument('--learning-rate', type=float,
            dest='initial_lr', help='learning rate (default %(default)s)',
            metavar='INITIAL_LR', default=INITIAL_LR)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)   
    return parser


parser = build_parser()
options = parser.parse_args()
train.train(options)
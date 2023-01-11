#!/usr/bin/env python3
from typing import List
from spreadsheet2 import SpreadSheet
from itertools import product
from argparse import ArgumentParser, RawTextHelpFormatter
from ninlib import EXPERIMENTAL_ID, MAIN, SUBJECT_ID, as_id_list, as_int, find_pair, CARD_ID
from logging import getLogger, basicConfig, INFO, WARNING
from sys import stdout, stdin
import csv

program_name = 'mergecsv'
parser = ArgumentParser(description=f'''
Merge two csv files.
''', formatter_class=RawTextHelpFormatter)

if stdin.isatty():
    parser.add_argument(
        'filename',
        type=str,
        help='''File name of csv.''')

parser.add_argument(
    'filename2',
    type=str,
    help='''File name of second csv.''')

parser.add_argument(
    '-v',
    '--verbose',
    action='store_true',
    help='''Show information of processing''')

parser.add_argument(
    'label',
    type=str,
    default=None,
    help='''Label of data.''')

parser.add_argument(
    '-l',
    '--label2',
    type=str,
    default=None,
    help='''Label of data if differ.''')

parser.add_argument(
    '-o',
    '--output',
    type=str,
    default=None,
    help='''File name to output.
default is 'out.csv'.''')

parser.add_argument(
    '--enc',
    type=str,
    default='cp932',
    help='''Name of encoding.[shift_jis, utf_8]''')


args = parser.parse_args()
basicConfig(level = INFO if args.verbose else WARNING)
logger = getLogger(program_name)
logger.info(f'( ･`ω･´)< Start running...{program_name}')

# Read all the csv
with open(args.filename2) as fp:
    data_to_add = SpreadSheet().load_data(fp.readlines(), encoding=args.enc)
data_list: SpreadSheet
if stdin.isatty():
    with open(args.filename) as fp:
        data_list = SpreadSheet().load_data(fp.readlines(), encoding=args.enc)
else:
    data_list = SpreadSheet().load_data(stdin.readlines(), encoding=args.enc)

if args.label2:
    result = data_list.concat(data_to_add, label=args.label2, anotherlabel=args.label)
else:
    result = data_list.concat(data_to_add, label=args.label)
result.to_csv()

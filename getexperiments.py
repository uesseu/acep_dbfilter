#!/usr/bin/env python
from typing import Callable, Optional, List, Union, cast, Iterable, Any
from spreadsheet import SpreadSheet
from argparse import ArgumentParser, RawTextHelpFormatter
from ninlib import EXPERIMENTAL_ID, MAIN, SUBJECT_ID, as_id_list, as_int, find_pair, CARD_ID
from logging import getLogger, basicConfig, INFO, WARNING

program_name = 'subj2exp'
parser = ArgumentParser(description=f'''
Get csv file filtered by expriment id.
It needs csv file from the database.
The csv file needs to involve experiments id and subject id.

It returns
    >> {program_name} experiment.csv experiment.csv -o experiments.csv
''', formatter_class=RawTextHelpFormatter)
parser.add_argument(
    'filename',
    type=str,
    help='''File name of csv.
It is regular csv from database.''')

parser.add_argument(
    'id_list',
    type=str,
    help='''File name of subjects csv.
experiments id file is just numerics separated by lines like ...

1
4
23
156
''')

parser.add_argument(
    '--enc',
    type=str,
    default='shift-jis',
    help='''Name of encoding.[shift_jis, utf_8]''')

parser.add_argument(
    '-o',
    '--output',
    type=str,
    default='experiments.csv',
    help='''File name to output.
Default is 'experiments.csv'.''')
parser.add_argument(
    '-l',
    '--log',
    action='store_true',
    help='''Show information of processing''')

args = parser.parse_args()
basicConfig(level = INFO if args.log else WARNING)
logger = getLogger(program_name)
logger.info(f'( ･`ω･´)< Start running...')

data_list = SpreadSheet().load_csv(args.filename, encoding=args.enc)
with open(args.id_list) as fp:
    ids = [x.strip() for x in fp.readlines()]
data_list.isin(EXPERIMENTAL_ID, ids).to_csv(args.output)

logger.info(f'''(*´∀｀*)< Done! "{args.output}" was written.''')

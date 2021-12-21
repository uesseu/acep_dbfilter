from pathlib import Path
from collections import OrderedDict
from typing import Callable, Optional, List, Union, cast, Iterable, Any
from spreadsheet import SpreadSheet, xlsx2csv
from nintimeit import TimeIt
from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from ninlib import EXPERIMENTAL_ID, MAIN, SUBJECT_ID, as_id_list, as_int, find_pair, CARD_ID
from logging import getLogger, basicConfig, INFO, WARNING
import csv

program_name = 'subj2exp'
parser = ArgumentParser(description='''
Extract examination id by subject id.
It needs csv file from the database.
The csv file needs to involve experiments id and subject id.
It returns
    >> {program_name} subjects.csv subject_ids.csv -o experiments.csv
''', formatter_class=RawTextHelpFormatter)
parser.add_argument(
    'filename',
    type=str,
    help='File name of csv.')
parser.add_argument(
    'id_list',
    type=str,
    help='File name of subjects csv.')
parser.add_argument(
    '-o',
    '--output',
    type=str,
    default='experiments_id.csv',
    help='File name to output.')
parser.add_argument(
    '-v',
    '--verbose',
    type=str,
    help='''File name of drug csv.
It is drug file from database.''')
args = parser.parse_args()
if args.verbose:
    basicConfig(level = INFO)
else:
    basicConfig(level = WARNING)
logger = getLogger(program_name)
logger.info(f'( ･`ω･´)< Start running...')


data_list = SpreadSheet().load_csv(args.filename, encoding='shift_jis')
with open(args.id_list) as fp:
    ids = [x.strip() for x in fp.readlines()]
result = data_list.isin(SUBJECT_ID, ids)[EXPERIMENTAL_ID]
with open(args.output, 'w') as fp:
    for r in result:
        fp.write(r + '\n')

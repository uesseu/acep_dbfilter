#!/usr/bin/env python3
from typing import List
from spreadsheet2 import SpreadSheet
from argparse import ArgumentParser, RawTextHelpFormatter
from ninlib import EXPERIMENTAL_ID, MAIN, SUBJECT_ID, as_int, SEP, DAY
from ninpipe import Pipe
from logging import getLogger, basicConfig, INFO, WARNING
from datetime import datetime
from sys import stdin, stderr
from textio import TextIO, AsyncRun

program_name = 'mergepsycho'
parser = ArgumentParser(description=f'''
Merge drug information csv file.
It needs csv file from the database.
The csv file needs to involve experiments id and subject id.
    >> {program_name} drugs.csv experiment.csv
    >> echo experiment.csv | {program_name} drugs.csv
''', formatter_class=RawTextHelpFormatter)

parser.add_argument(
    'psycho',
    type=str,
    help='''File name of drug csv.
It is psychological test file from database.''')

parser.add_argument(
    'filename',
    nargs='?',
    type=str,
    default='-',
    help='''File name of experiment csv.
It is regular experimental csv from database.''')

parser.add_argument(
    '-v',
    '--verbose',
    action='store_true',
    help='''Show information of processing''')

parser.add_argument(
    '-l',
    '--label',
    type=str,
    default=None,
    help='''Label of data.''')

parser.add_argument(
    '-b',
    '--before',
    type=int,
    default=30,
    help='''The day before experiment.
If it is not be set, 30 will be used.''')

parser.add_argument(
    '-a',
    '--after',
    type=int,
    default=30,
    help='''The day after experiment.
If it is not be set, 30 will be used.''')

parser.add_argument(
    '--enc',
    type=str,
    default='cp932',
    help='''Name of encoding.[shift_jis, utf_8_sig]''')


args = parser.parse_args()
basicConfig(level = INFO if args.verbose else WARNING)
logger = getLogger(program_name)
logger.info(f'( ･`ω･´)< Start running...{program_name}')


def make_day(day_string: str) -> datetime:
    sep = '/'
    if '.' in day_string:
        sep = '.'
    try:
        return datetime(*(as_int(x) for x in day_string.split(sep)))
    except BaseException:
        return datetime(1973, 1, 1)


def main():
    # Read all the csv
    psycho_raw = AsyncRun(TextIO(args.psycho).load_lines)()
    if args.filename != '-':
        with open(args.filename) as fp:
            data_list = SpreadSheet().load_data(fp.readlines())
    else:
        data_list = SpreadSheet().load_data(stdin.readlines())
    psychos = SpreadSheet().load_data(psycho_raw.get())


    result = SpreadSheet()
    errors: List[int] = []

    print(data_list)
    for data in data_list:
        subject = psychos[psychos[MAIN+SUBJECT_ID] == data[SUBJECT_ID]]
        experiment_day = make_day(data[DAY])
        psycho_tests = subject.filter(
            EXPERIMENTAL_ID+SEP+DAY,
            lambda x: -args.before
            < (make_day(x) - experiment_day).days
            < args.after
        ).calc()
        psycho_tests.set_label(args.label)
        if len(psycho_tests):
            test_dict = psycho_tests[0]
            data.update(test_dict)
            result.add_dict(data)
        else:
            result.add_dict(data)
            errors.append([data[EXPERIMENTAL_ID], data[SUBJECT_ID]])
    for error in errors:
        stderr.write(f'{program_name} could not get {error[0]} of {error[1]} from {args.psycho}\n')

    result.to_csv(encoding=args.enc)
    logger.info(f'''(*´∀｀*)< {program_name} has merged {args.psycho}!''')
main()

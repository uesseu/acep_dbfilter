#!/usr/bin/env python3
from .spreadsheet2 import SpreadSheet
from copy import deepcopy
from .textio import TextIO, AsyncRun
from argparse import ArgumentParser, RawTextHelpFormatter
from .ninlib import EXPERIMENTAL_ID
from logging import getLogger, basicConfig, INFO, WARNING
from sys import stdin, stderr
import time


class TimeIt:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        self.t = time.time()

    def __exit__(self, a, b, c) -> None:
        print(time.time() - self.t)


program_name = 'mergedrug'
parser = ArgumentParser(description=f'''
Merge drug information csv file.
It needs csv file from the database.
The csv file needs to involve experiments id and subject id.
    >> {program_name} drugs.csv experiment.csv
    >> cat experiment.csv | {program_name} drugs.csv
''', formatter_class=RawTextHelpFormatter)

parser.add_argument(
    'drugfile',
    type=str,
    help='''File name of drug csv.
It is drug file from database.''')

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
    help='''Show information of processing.
    If it is configured, stdin involves information.''')

parser.add_argument(
    '--enc',
    type=str,
    default='utf_8',
    help='''Name of encoding.[cp932, shift_jis, utf_8]''')

args = parser.parse_args()
basicConfig(level=INFO if args.verbose else WARNING)
logger = getLogger(program_name)
logger.info(f'( ･`ω･´)< Start running...{program_name}')


def main():
    # Read all the csv
    drag_thread = AsyncRun(TextIO(args.drugfile).load_lines)()

    if args.filename != '-':
        data_list = SpreadSheet().load_data(TextIO(args.filename).load_lines())
    else:
        data_list = SpreadSheet().load_data(stdin.readlines())
    experiments = [i for i in data_list[EXPERIMENTAL_ID] if i != '']

    def select_experiments(id_in_drug: str) -> str:
        for ex_id in experiments:
            for drug_id in id_in_drug.split(','):
                if drug_id.strip() == ex_id.strip():
                    return ex_id
        return ''

    drugs = SpreadSheet(mutable=True).load_data(drag_thread.get())
    tmp_experiments = deepcopy(experiments)

    def select_only_exp(x):
        for n, e in enumerate(tmp_experiments):
            if e in x.split(','):
                del tmp_experiments[n]
                return True
        return False

    def is_not_empty(x):
        return x != ''

    drugs = drugs.filter(EXPERIMENTAL_ID, select_only_exp)\
        .map(EXPERIMENTAL_ID, select_experiments)\
        .filter(EXPERIMENTAL_ID, is_not_empty).calc()

    drugs.set_label('drugs')
    result = SpreadSheet()

    res: dict

    def is_ex(x):
        return x == ex

    for ex in experiments:
        try:
            dr = drugs.filter(EXPERIMENTAL_ID, is_ex).calc()
            da = data_list.filter(EXPERIMENTAL_ID, is_ex).calc()
            if len(dr) != 0:
                tmp_dr = da[0]
                tmp_dr.update(dr[0])
                result.add_dict(tmp_dr)
            else:
                tmp_dr = da[0]
                result.add_dict(tmp_dr)
        except IndexError as er:
            raise er
        except BaseException as er:
            raise er
    result.to_csv(None, encoding=args.enc)
    logger.info(f'''(*´∀｀*)< {program_name} has Done!''')

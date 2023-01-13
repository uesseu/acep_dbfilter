#!/usr/bin/env python3
from spreadsheet2 import SpreadSheet
from textio import TextIO
from argparse import ArgumentParser, RawTextHelpFormatter
from ninlib import EXPERIMENTAL_ID
from sys import stdin

program_name = 'shrinkdrug'
parser = ArgumentParser(description=f'''
Shrink drug information.
By this script, drug information from kyushu university becomes small.
It needs csv file of drug information from the database.
    >> {program_name} drugs.csv
    >> cat drugs.csv | {program_name}
''', formatter_class=RawTextHelpFormatter)

parser.add_argument(
    'drugfile',
    nargs='?',
    type=str,
    default='-',
    help='''File name of drug csv.
It is drug file from database.''')

parser.add_argument(
    '--enc',
    type=str,
    default='utf_8',
    help='''Name of encoding.[cp932, shift_jis, utf_8]''')

args = parser.parse_args()
NULL_ID = ['0,,0', '00,,00']


def main():
    if args.drugfile != '-':
        data_list = SpreadSheet().load_data(TextIO(args.drugfile).load_lines())
    else:
        data_list = SpreadSheet().load_data(stdin.readlines())

    data_list.filter(EXPERIMENTAL_ID, lambda x: x not in NULL_ID)\
        .to_csv(None, encoding=args.enc)
main()

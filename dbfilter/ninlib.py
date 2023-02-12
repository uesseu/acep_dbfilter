from typing import List
from sys import stderr
from spreadsheet2 import SpreadSheet
EXPERIMENTAL_ID = '受診日ID'
OLD_EXPERIMENTAL_ID = '実験固有ID'
SUBJECT_ID = 'ID'
CARD_ID = 'カルテID'
SEP = ' ::'
MAIN = 'DGDメイン::'
DAY = '実験日'


def as_int(string: str) -> int:
    try:
        return int(float(string))
    except:
        return 0


def as_id_list(string: str) -> List[int]:
    tmp_numbers = [as_int(s) for s in string.split(',')]
    return [s for s in tmp_numbers if s != 0]


def find_pair(texts: list, targets: list) -> bool:
    return any(text in targets for text in texts)


class IDConstructor:
    def __init__(self) -> None:
        self.inst: str = ''
        self.inst_length: int = 0

    def set_inst(self, label: str) -> 'IDConstructor':
        self.inst = label
        self.inst_length = len(label)
        return self

    def has_instid(self) -> bool:
        return self.inst != ''

    def get_inst_id(self, sp: SpreadSheet, label: str) -> 'IDConstructor':
        '''Returns univ label
        sp: SpreadSheet
            SpreadSheet of institute.
        '''
        subj_str = list(sp[label].data)[0]
        self.set_inst(subj_str.split('_')[0] if '_' in subj_str else '')
        return self

    def add_inst_id(self, num: str) -> str:
        '''Add inst to id'''
        return '_'.join((self.inst, num)) if '_' not in num else num

    def remove_inst_id(self, id_str: str) -> str:
        if id_str[:self.inst_length] == self.inst:
            return id_str[self.inst_length+1:]
        else:
            return id_str


def main():
    print('test')
    idc = IDConstructor()
    idc.set_inst('kyushu')
    assert idc.remove_inst_id('kyushu_4') == '4'
    assert idc.add_inst_id('4') == 'kyushu_4'

if __name__ == '__main__':
    main()

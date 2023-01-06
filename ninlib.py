from typing import List
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

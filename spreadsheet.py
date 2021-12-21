from typing import (Callable, Optional, List, Union, cast,
                    Iterable, Any, Generator, Iterator)
from pathlib import Path
from copy import deepcopy
import csv
import operator
from enum import Enum
from itertools import tee
try:
    import numpy as np
except:
    class np:
        def array(self): pass
        class ndarray: pass

class ListLikeIterator:
    def __init__(self, data: Optional[Iterable] = None):
        self.data: List[Iterator] = [] if data is None else [iter(data)]

    def __iter__(self) -> 'ListLikeIterator':
        self._iter_num = 0
        return self

    def __next__(self) -> Any:
        try:
            result = self.data[self._iter_num].__next__()
        except StopIteration as er:
            self._iter_num += 1
            if self._iter_num == len(self.data):
                raise StopIteration()
            result = self.data[self._iter_num].__next__()
        return result

    def __add__(self, data: Iterable) -> 'ListLikeIterator':
        inner: List[Iterator] = []
        outer: List[Iterator] = []
        for d in self.data:
            tmp_in, tmp_out = tee(d)
            inner.append(tmp_in)
            outer.append(tmp_out)
        self.data = inner
        result = ListLikeIterator()
        result.data = outer + [iter(data)]
        return result

    def append(self, data: Iterable) -> None:
        self.data.append(iter(data))

def join_as_csv(texts: Iterable[str]) -> str:
    return ','.join(f'"{text}"' for text in texts) + '\n'

def set_text_length(text: str, length: int) -> str:
    text_length = len(text)
    if text_length < length:
        text = text + ' ' * (length - len(text))
    elif text_length == length:
        pass
    else:
        text = text[:length]
    return text

def as_str(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    elif isinstance(x, np.ndarray):
        return f'Numpy{x.shape}'.strip()
    elif hasattr(x, '__iter__'):
        return str(type(x))
    return str(x).strip()

class Column:
    def __init__(self, data: Iterable, id: int, label: str, has_list: bool = False):
        self.data = data
        self.id = id
        self.label = label
        self.oper: Callable[[Any, Any], bool]
        self.has_list = has_list

    def __lt__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.lt
        self.target = target
        return (x < target for x in self.data)

    def __le__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.le
        self.target = target
        return (x <= target for x in self.data)

    def __ge__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.ge
        self.target = target
        return (x >= target for x in self.data)

    def __gt__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.gt
        self.target = target
        return (x > target for x in self.data)

    def __eq__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.eq
        self.target = target
        return (x == target for x in self.data)

    def __ne__(self, target: Union[int, float]) -> Iterable[bool]:
        self.oper = operator.ne
        self.target = target
        return (x != target for x in self.data)

    def __iter__(self) -> Iterable:
        return self.data.__iter__()

    def __add__(self, column: 'Column') -> 'Column':
        result = deepcopy(self)
        result.data = list(self.data) + list(column.data)
        return result

    def __str__(self) -> str:
        if not self.has_list:
            self.data = list(self.data)
        return str(self.data)

    def __len__(self) -> int:
        if not self.has_list:
            self.data = list(self.data)
        return len(cast(list, self.data))

    def __and__(self, target: Union[Iterable, 'Column']) -> 'Column':
        if isinstance(target, Column):
            target = target().data
        elif isinstance(target, list):
            pass
        else:
            target = list(target)
        self()
        return Column(cast(list, self.data)+cast(list, target),
                      id(self), self.label, True)

    def __call__(self) -> 'Column':
        if not self.has_list:
            self.data = list(self.data)
            self.has_list = True
        return self


class IterType(Enum):
    list = 0
    iterator = 1
    numpy = 2
    tuple = 3


def as_itertype(obj: Union[np.ndarray, tuple, list, Iterable],
                iter_type: IterType) -> Union[Iterable, np.ndarray]:
    if iter_type == IterType.list:
        return list(obj)
    elif iter_type == IterType.iterator:
        return obj
    elif iter_type == IterType.numpy:
        return np.array(tuple(obj))
    elif iter_type == IterType.tuple:
        return tuple(obj)
    return []


class SpreadSheet:
    ''' SpreadSheet like object.
    It reads list of ordered dict, and perform like spread sheet.

    data: List[OrderedDict]
        Data of spread sheet.
    '''
    def __init__(self, index: list = [],
                 data: Union[list, tuple, ListLikeIterator, np.array] = [],
                 name: str = '',
                 iter_type: IterType = IterType.tuple) -> None:
        '''
        data: Optional[List[OrderedDict]] = None
            A list of ordered dict to manage.
        '''
        self.data = data
        self.index_keys = index
        self.index = dict(((value, num) for num, value in enumerate(index)))
        self.name = name
        self.type = iter_type
        if self.type == IterType.iterator:
            self.data = ListLikeIterator(data)

    def load_csv(self, fname: str,
                 index: Optional[list] = None,
                 encoding: str='utf_8') -> 'SpreadSheet':
        ''' Read csv file.
        fname: str or Path
            Path of csv.
        index: 
            Labels of csv.
            If it is None, first line of csv will be read as label.
        '''
        with open(fname, encoding=encoding) as fp:
            text = fp.readlines()
        data = list(csv.reader(text, dialect='excel'))
        self.index_keys = index if index else data.pop(0)
        self.index = dict(((value, num) for num, value
                           in enumerate(self.index_keys)))
        self.data = as_itertype(data, self.type)
        return self

    def __getitem__(self,
                    args: Union[None, Iterable, str,
                                int, Column, tuple]) -> Any:
        self.calc()
        self.data = cast(list, self.data)
        if isinstance(args, slice):
            return Column(
                self.data[args],
                id(self),
                args)
        if isinstance(args, tuple):
            row = args[0]
            col = args[1]
            if isinstance(row, slice):
                return self.data[row][self.index[args[1]]]
            return self.data[args[0]][self.index[args[1]]]
        elif self.data is None:
            return []
        elif isinstance(args, Column):
            def wrap(x: Any) -> bool:
                args = cast(Column, args)
                return args.oper(x, args.target)
            return self.filter(args.label, wrap)
        elif hasattr(args, '__iter__') and not isinstance(args, str):
            args = cast(Iterable, args)
            return SpreadSheet(
                self.index_keys,
                self.data[args] if isinstance(self.data, np.ndarray)
                else [d for d, a in zip(self.data, args) if a],
                self.name, self.type
            )
        elif isinstance(args, int):
            return dict(zip(self.index, self.data[args]))
        elif isinstance(args, str):
            return Column(
                self.data[:, self.index[args]]
                if isinstance(self.data, np.ndarray)
                else (d[self.index[args]] for d in self.data),
                id(self),
                args)

    def calc(self) -> 'SpreadSheet':
        '''
        This object contains data as generators or iterators
        if it is possible to reduce memory.
        This method make the data to list.
        '''
        # if not isinstance(self.data, (list, np.ndarray)):
        if self.type == IterType.list and not isinstance(self.data, list):
            self.data = list(self.data)
        elif self.type == IterType.numpy and not isinstance(self.data, np.ndarray):
            self.data = np.array(tuple(self.data))
        elif self.type == IterType.tuple and not isinstance(self.data, tuple):
            self.data = tuple(self.data)
        elif self.type == IterType.iterator:
            self.data = tuple(self.data)
        return self

    def __str__(self) -> str:
        self.calc()
        if not hasattr(self.data, '__len__'):
            return 'Not calculated yet'
        else:
            if len(cast(list, self.data)) > 5:
                print('Too big data, Not all display')
                max_len = 5
            else:
                max_len = len(cast(list, self.data))
            texts :List[List[str]] = []
            length: List[int] = []
            result: List[str] = []
            texts.append(self.index_keys)
            for d in cast(list, self.data)[:max_len]:
                texts.append(d)
            for n, i in enumerate(self.index_keys):
                length.append(max(len(as_str(t[n])) for t in texts))
            result.append(' '.join((set_text_length(as_str(i), length[n])
                                    for n, i in enumerate(self.index_keys))))
            for d in cast(list, self.data)[:max_len]:
                result.append(' '.join(
                    (set_text_length(as_str(x), length[n])
                     for n, x in enumerate(d))))
            return '\n'.join(result)

    def filter(self, label: str,
               func: Callable[[Any], bool]) -> 'SpreadSheet':
        ''' Filter the data by function
        func: function to filter
        ----------
        Returns
        new SpreadSheet
        '''
        index = self.index[label]
        def wrap(d: list) -> bool:
            return func(d[index])
        return SpreadSheet(
            self.index_keys,
            self.data[func(self.data)] if isinstance(self.data, np.ndarray)
            else filter(wrap, self.data),
            self.name, self.type)


    def isin(self, label: str, target: Iterable) -> 'SpreadSheet':
        index = self.index[label]
        def wrap(d: list) -> bool: return d[index] in target
        return SpreadSheet(self.index_keys, filter(wrap, self.data),
                           self.name, self.type)


    def map(self, labels: Union[Iterable[str], str],
            func: Union[Iterable[Callable], Callable]) -> 'SpreadSheet':
        ''' convert labels by function
        labels: Union[str, int, List[str]]
            List of labels to convert.
            If it is string or int, only one column will be converted.
        func: Union[Callable, List[Callable]]
            Function to convert data.
            If it is list, apply functions for all of labels.
            If it is not list, func will be applied for all of 'labels'.
        ----------
        Returns
        new SpreadSheet
        '''
        self.calc()
        if self.data is None:
            return SpreadSheet()
        data = deepcopy(self.data)
        if isinstance(labels, str):
            index = self.index[labels]
            for d in data:
                d[index] = cast(Callable, func)(d[index])
        elif hasattr(func, '__iter__'):
            indices = [self.index[label] for label in labels]
            for d in data:
                for index, one_func in zip(indices, cast(Iterable, func)):
                    d[index] = one_func(d[index])
        else:
            indices = [self.index[label] for label in labels]
            for d in data:
                results = cast(Callable, func)(
                    *(d[index] for index in indices))
                for index, result in zip(indices, results):
                    d[index] = result
        return SpreadSheet(self.index_keys, data, self.name, self.type)

    def __len__(self) -> int:
        self.calc()
        return len(cast(list, self.data))

    def add_column(self, new_label: str, func: Callable) -> 'SpreadSheet':
        '''
        Add new label based on column by function.

        new_label: str
            Name of new label.
        func:
            Function to make new label.
            It gets dictionary and returns value.
        '''
        self.calc()
        self.index_keys.append(new_label)
        self.index = dict(((value, num) for num, value
                           in enumerate(self.index_keys)))
        for data in self.data:
            data.append(func(dict(zip(self.index_keys, data))))
        return self

    def make_dict(self, label: str, element: Optional[str] = None) -> dict:
        '''
        Convert itself as dictionary.
        Values is iterator of data.

        label: str
            Keys of dictionary.
        element: Optional[str]
            Value of dictionary.
            If it is None, all the items will be contained.
        '''
        return dict(zip(self[label], self[element] if element else self.data))

    def concat(self, label: str, spreadsheet: 'SpreadSheet') -> 'SpreadSheet':
        '''
        Concatnates two spread sheets by label.

        label: str
            Label name to concatnate.
        spreadsheet: SpreadSheet
            SpreadSheet object to append.
        '''
        base = deepcopy(self)
        header = base.name + ':' if base.name else ''
        header2 = spreadsheet.name + ':' if base.name else ''

        spre_dict = spreadsheet.make_dict(label)
        base_dict = base.make_dict(label)
        base_data = [base_dict[l] + spre_dict[l]
                     for l in base_dict.keys()
                     if l in spre_dict]

        base.index_keys = [header + b for b in base.index_keys]
        base.index_keys += [header2+i if i in base.index_keys
                       else header2+i+'_'
                       for i in spreadsheet.index_keys]
        base.index = dict(((value, num) for num, value
                           in enumerate(base.index_keys)))
        base.data = [b+s for b, s in zip(base.data, spreadsheet.data)]
        return base

    def to_csv(self, fname: Union[None, Path, str] = None, encoding: str = 'utf_8') -> None:
        '''
        Write csv.
        If fname is None, just print it.

        fname: Union[Path, str]
            File name of csv.
        '''
        if fname is None:
            print(join_as_csv(self.index_keys))
            for data in self.data:
                print(join_as_csv(data))
        else:
            with open(fname, 'w', encoding=encoding) as fp:
                fp.write(join_as_csv(self.index_keys))
                fp.writelines(join_as_csv(data)
                              for data in self.data)

    def add_dict(self, data: dict) -> None:
        '''
        Add dictionary to the spreadsheet.
        It makes new keys if spreadsheet has no key same as data has.

        data: dict
            Data to add.
        '''
        self.calc()
        container = [None] * len(self.index_keys)
        unknown_index = []
        for n in data.keys():
            if n in self.index:
                container[self.index_keys.index(n)] = data[n]
            else:
                unknown_index.append(n)
        self.index_keys += unknown_index
        self.data = [d + [None] * len(unknown_index)
                     for d in self.data]
        self.data.append(container + [data[u] for u in unknown_index])
        self.index = dict(((value, num) for num, value
                           in enumerate(self.index_keys)))

    def __add__(self, spreadsheet: 'SpreadSheet') -> 'SpreadSheet':
        '''
        Concatenate two spreadsheets which has same index.
        '''
        if self.type != spreadsheet.type:
            raise TypeError(
                f'Two spreadsheet {self.type} and '
                f'{spreadsheet.type} must be same'
            )
        if self.type == IterType.iterator:
            result = deepcopy(self)
            result.data = result.data + spreadsheet.data
            return result
        self.calc()
        spreadsheet.calc()
        if len(self.index) != len(spreadsheet.index):
            raise BaseException('Not same index')
        for m, n in zip(self.index, spreadsheet.index):
            if m != n:
                raise BaseException('Not same index')
        result = deepcopy(self)
        result.data += spreadsheet.data
        return result

    def _check_type(self, not_iterator: bool = False,
                    has_len: bool = False) -> None:
        if not_iterator and self.type == IterType.iterator:
            TypeError('Iter type must not be iterator.')
        if has_len and not hasattr(self, '__len__'):
            TypeError('Iter type has no __len__.')

    def delete_column(self, label: str) -> 'SpreadSheet':
        '''
        Delete column as a mutable object.
        '''
        self._check_type(not_iterator=True)
        index = self.index_keys.index(label)
        self.index_keys.pop(index)
        self.index.pop(label)
        for data in self.data:
            data.pop(index)
        return self

    def melt(self, labels: List[str],
             new_label: str, new_data: str) -> 'SpreadSheet':
        '''
        Melt the spreadsheet.

        labels: List[str]
            Labels to melt.
        new_label: str
            Name of column which contains label name.
        new_data: str
            Name of column whihc conitains data.
        '''
        self.calc()
        frag = deepcopy(self)
        fragments = []
        for label in labels:
            frag.delete_column(label)
        for label in labels:
            tmp = deepcopy(frag)
            for t, data in zip(tmp.data, self[label]):
                t.append(label)
                t.append(data)
            fragments.append(tmp)
        result = deepcopy(frag)
        result.data = []
        for frag in fragments:
            result.data += frag.data
        result.index_keys.append(new_label)
        result.index_keys.append(new_data)
        result.index = dict(((value, num) for num, value
                           in enumerate(result.index_keys)))
        return result

    def split_label(self, label: str, new_labels: List[str],
                    sep: str) -> 'SpreadSheet':
        '''
        Split melted label.
        At first, melt method melt the spreadsheet,
        but it is not easy to use.
        If the original label is splitted by any character,
        like camel case, you can split it to multiple columns.

        label: str
            Name of column to split.
        new_labels: List[str]
            Names of new columns.
        sep: str
            Separator of string.
        '''
        result = deepcopy(self)
        index = result.index[label]
        result.index_keys += new_labels
        for data in result.data:
            data += data[index].split(sep)
        result.index = dict(((value, num) for num, value
                           in enumerate(result.index_keys)))
        return result

    def __iter__(self) -> 'SpreadSheet':
        self._iter_num = 0
        self.calc()
        self._iter_max = len(cast(list, self.data))
        return self

    def __next__(self) -> dict:
        result = cast(dict, self[self._iter_num])
        self._iter_num += 1
        if self._iter_max == self._iter_num:
            raise StopIteration()
        return result

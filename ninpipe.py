#!usr/bin/env python3
from sys import stdin, stdout, stderr
from typing import List, Callable, Iterable, Optional
from argparse import ArgumentParser
from threading import Thread
from subprocess import PIPE, run, Popen
from sys import stdout, stdin

class PipeState:
    object: Optional['PipeState'] = None

    def __new__(cls) -> 'PipeState':
        if PipeState.object is None:
            PipeState.object = object.__new__(cls)
        return PipeState.object

    def __init__(self) -> None:
        self.from_pipe = False if stdin.isatty() else True
        self.to_pipe = False if stdout.isatty() else True


pipestate = PipeState()


class Pipe:
    '''
    Class to use pipe.
    It is iterable and read lines by 'for statement'.
    Further more, it can read lines asynchronously.
    Asynchronous method is async_iter().
    '''
    def __init__(self, sep: str = '\n') -> None:
        '''
        sep: str
            Separator of each lines.
            It may be '\n' if libe based.
            Multiple characters cannot be contained.
        '''
        self.sep = sep
        self.ended = False
        self.num = 0

    def get_all_lines(self) -> List[str]:
        '''
        Get all the stdin.
        It is not good for performance.
        '''
        text = []
        while(True):
            try:
                text.append(input())
            except:
                break
        return text

    def get(self, num: int = 0) -> str:
        '''
        Get one line stdin.

        num: int = 0
            Length of string to read.
            If it is 0, string will be read until 'sep' was read.
        '''
        if self.ended:
            raise EOFError
        if num != 0:
            return stdin.read(num)
        string: List[str] = []
        while True:
            data = stdin.read(1)
            if data == '':
                self.ended = True
                self.result = ''.join(string)
                return self.result
            elif data == self.sep:
                self.result = ''.join(string)
                return self.result
            else:
                string.append(data)

    def __next__(self) -> str:
        if self.ended:
            raise StopIteration
        result = self.get()
        if result == '':
            raise StopIteration
        return result

    def __iter__(self) -> 'Pipe':
        return self

    def async_iter(self) -> 'AsyncPipe':
        '''
        Asynchronous version.
        Because it cannot detect EOF asynchronously, you must detect EOF.
        If it is blank string, EOF was arrived.

        >>> for data in Pipe().async_iter():
        >>>     received = data.receive()
        >>>     if reveived:
        >>>         print(reveived)
        '''
        return AsyncPipe(self)

    def put(self, data: str) -> None:
        '''
        Write data to stdout.
        '''
        stdout.write(data)

class AsyncPipe:
    def __init__(self, pipe: Pipe) -> None:
        self.pipe = pipe

    def request(self, num: int = 0) -> 'AsyncPipe':
        self.reading = Thread(target=self.pipe.get, args=(num,))
        self.reading.start()
        self.now_reading = True
        return self

    def receive(self) -> str:
        if self.now_reading:
            self.reading.join()
        return self.pipe.result

    def __next__(self) -> 'AsyncPipe':
        if self.pipe.ended:
            raise StopIteration
        return self.request()

    def __iter__(self) -> 'AsyncPipe':
        return self

def main() -> None:
    for n in Pipe().async_iter():
        if not n.receive():
            break
        print(n.receive(),  'h')
    

if __name__ == '__main__':
    main()


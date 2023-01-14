#!usr/bin/env python3
import sys
from sys import stdin, stdout
from typing import List, Optional
from argparse import ArgumentParser
from threading import Thread
from subprocess import run, PIPE


class PipeState:
    '''Singleton representing the state of the pipe'''
    object: Optional['PipeState'] = None

    def __new__(cls) -> 'PipeState':
        if PipeState.object is None:
            PipeState.object = object.__new__(cls)
        return PipeState.object

    def __init__(self) -> None:
        self.from_pipe = False if stdin.isatty() else True
        self.to_pipe = False if stdout.isatty() else True


pipestate = PipeState()


def switch_pipe(fname_from_args: str,
                args: Optional[ArgumentParser] = None) -> List[str]:
    '''
    Switch pipe mode by tty or not and returns list of strings.
    '''
    fname = None
    if args and hasattr(args, fname_from_args):
        fname = getattr(args, fname_from_args)
    if stdin.isatty() and fname:
        with open(fname) as fp:
            return fp.readlines()
    else:
        return Pipe().get_all_lines()


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
        return stdin.readlines()

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
        '''
        Request something by pipe asynchronously.
        '''
        self.reading = Thread(target=self.pipe.get, args=(num,))
        self.reading.start()
        self.now_reading = True
        return self

    def receive(self) -> str:
        """
        Receive something by pipe asynchronously.

        If you want to use this in for statement,
        it needs to set end. It runs asynchronously,
        it is impossible to detect EOF by itself.
        And so, write like this.
        >>> for n in Pipe().async_iter():
        >>>     if not n.receive():
        >>>         break
        >>>     print(n.receive(), 'h')
        """
        if self.now_reading:
            self.reading.join()
        return self.pipe.result

    def __next__(self) -> 'AsyncPipe':
        if self.pipe.ended:
            raise StopIteration
        return self.request()

    def __iter__(self) -> 'AsyncPipe':
        return self


class AsyncRun:
    pass


def main() -> None:
    for n in Pipe().async_iter():
        if not n.receive():
            break
        print(n.receive(),  'h')


if __name__ == '__main__':
    main()

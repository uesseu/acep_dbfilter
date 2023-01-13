from typing import Union, Callable, Optional, Any
from pathlib import Path
from threading import Thread
PathLike = Union[str, Path]


class AsyncRun:
    def __init__(self, func: Callable):
        self.func = func

        def wrap(*args, **kwargs) -> None:
            self.result = self.func(*args, **kwargs)
        self.result: Optional[Any] = None
        self.thread = None
        self.wrap = wrap

    def __call__(self, *args, **kwargs) -> 'AsyncRun':
        self.thread = Thread(target=self.wrap, args=args, kwargs=kwargs)
        self.thread.start()
        return self

    def get(self) -> Any:
        self.thread.join()
        return self.result


class TextIO:
    encodes = ['cp932', 'utf_8', 'utr_8_sig', 'shift_jis']

    def __init__(self, filename: PathLike):
        self.filename = filename
        self.was_failed = False

    def load(self) -> str:
        self.was_failed = True
        result: str
        data: bytes
        error: BaseException
        try:
            with open(self.filename, 'rb') as fp:
                data = fp.read()
        except BaseException as er:
            raise er
        for encoding in self.encodes:
            try:
                result = data.decode()
                self.was_failed = False
                return result
            except BaseException as er:
                error = er
        raise error

    def load_lines(self) -> str:
        return self.load().split('\n')

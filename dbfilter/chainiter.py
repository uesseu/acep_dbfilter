"""
Ninja speed iterator package.
This is an iterator like Array object of Node.js.
Chainable fast iterators can be used.
Furturemore, generator based pipeline can be used.
"""

# Chainiter is built on objects.
# ChainIterPrivate
# ChainIterBase
# ChainIterNormal
# ChainIterAsync
#
# Object hierarchie is...
# Private -> Base
# Base -> Normal
# Base -> Async
# Normal + Async -> ChainIter
from asyncio import new_event_loop, ensure_future, Future
from typing import (Any, Callable, Iterable, cast, Coroutine,
                    Union, Optional, Tuple,
                    Iterator, List)
from itertools import starmap
from multiprocessing import Pool
from functools import reduce, wraps, partial
from logging import (getLogger, Logger, NullHandler,
                     INFO, WARNING, ERROR, CRITICAL)
from inspect import _empty, signature
import time
from threading import Thread
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
logger = getLogger('chainiter')
logger.addHandler(NullHandler())


class ChainIterMode(Enum):
    SINGLE = 'single'
    THREAD = 'thread'
    PROCESS = 'process'
    BOTH = 3


mode = ChainIterMode


def separate_list(data, sep) -> List[list]:
    length = len(data)
    ssep = int(length / sep)
    result = [[n for n in data[ind:ind + ssep]]
              for ind in range(0, length, ssep)]
    return result


def compose(*funcs: Callable) -> Callable:
    """
    Just a compose function
    >>> def multest(x): return x * 2
    >>> compose(multest, multest)(4)
    16
    """
    def composed(*args: Any, **kwargs) -> Any:
        return reduce(lambda x, y: y(x),
                      (funcs[0](*args, **kwargs),) + funcs[1:])
    return composed


def _curry_one(func: Callable) -> Callable:
    """
    >>> def multest2(x, y): return x * y
    >>> fnc = _curry_one(multest2)
    >>> fnc(2)(3)
    6
    """
    def wrap(*args: Any, **kwargs: Any) -> Any:
        return partial(func, *args, **kwargs)
    return wrap


def curry(num_of_args: Optional[int] = None) -> Callable:
    """
    Just a curry function.
    num_of_args: Optional[int]
        Number of functions.
        It is detected automatically, but it can be set manually too.
        It can be used as decorator.
    >>> def multest2(x, y): return x * y
    >>> fnc = curry(2)(multest2)
    >>> fnc(2)(3)
    6
    >>> def multest3(x, y, z): return x * y * z
    >>> fnc = curry()(multest3)
    >>> fnc(2)(3)(4)
    24
    """
    def curry_wrap(func: Callable) -> Callable:
        wr = wraps(func)
        length_of = compose(filter, list, len)
        if num_of_args:
            num = num_of_args
        else:
            def is_empty(x: Any) -> bool: return x.default is _empty
            num = length_of(is_empty, signature(func).parameters.values())
        for n in range(num - 1):
            func = _curry_one(func)
        return wr(func)
    return curry_wrap


def write_info(func: Callable, chunk: int = 1,
               logger: Logger = logger, mode: str = 'single') -> None:
    """
    Log displayer of chainiter.
    """
    if hasattr(func, '__name__'):
        logger.info(' '.join(('Running', str(func.__name__))))
    else:
        logger.info('Running something without name.')
    if chunk > 1:
        logger.info(f'Computing by {chunk} {mode}!')


def _asit_wrap(data: list, n: int, func: Callable, *args, **kwargs):
    data[n] = func(data[n], *args, **kwargs)


def _asit_starwrap(data: list, d: Any, n: int, func: Callable,
                   *args, **kwargs):
    data[n] = func(*d, *args, **kwargs)


class AsyncIterator:
    '''
    Thread based async iterator.
    Because it is asynchoronious, it may be faster
    with functions for IO.
    Numpy operation may be fast too.

    ai = AsyncIterator(range(10), chunk=8)
    ai.map(func).get()
    '''
    def __init__(self, data: list, chunk: int = 1):
        self.data = list(data)
        self.chunk = chunk
        self.started = False

    def _start_map(self):
        self.length = len(list(self.data))
        self.async_q = [None] * self.length

    def starmap(self, func: Callable, chunk: int = 1, *args: list,
                **kwargs: Any) -> 'AsyncIterator':
        if self.started:
            self._async_starmap(func, *args, **kwargs)
        self._start_map()
        self.ths = [Thread(target=_asit_starwrap,
                           args=(self.data, d, num, func, *args),
                           kwargs=kwargs)
                    for num, d in enumerate(self.data)]
        [th.start() for th in self.ths]
        self.started = True
        return self

    def map(self, func: Callable, chunk: int = 1, *args,
            **kwargs: Any) -> 'AsyncIterator':
        if self.started:
            self._async_map(func, *args, **kwargs)
        self._start_map()
        self.ths = [Thread(target=_asit_wrap,
                           args=(self.data, num, func, *args),
                           kwargs=kwargs)
                    for num in range(len(self.data))]
        [th.start() for th in self.ths]
        self.started = True
        return self

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return self.get()

    def get(self) -> Any:
        if self.started:
            self.started = False
            [th.join() for th in self.ths]
            return self.data
        else:
            raise BaseException('Use start method before using this.')

    def is_completed(self) -> List[bool]:
        return [not t.is_alive() for t in self.ths]

    def _async_map(self, func: Callable, *args, **kwargs):
        while not all(self.is_completed()):
            for index, completed in enumerate(self.is_completed()):
                if completed:
                    self.data[index] = self.runner(
                        target=_asit_wrap,
                        args=(self.data, index, func, args),
                        kwargs=kwargs)

    def _async_starmap(self, func: Callable, *args, **kwargs):
        while not all(self.is_completed()):
            for index, completed in enumerate(self.is_completed()):
                if completed:
                    self.data[index] = self.runner(
                        target=_asit_wrap,
                        args=(self.data, index, func, *args),
                        kwargs=kwargs)


def run_coroutine(col: Coroutine) -> Any:
    loop = new_event_loop()
    result = loop.run_until_complete(col)
    loop.close()
    return result


def start_async(func: Callable, *args, **kwargs) -> Future:
    """
    Start async function instantly.
    """
    async def wrap():
        return func(*args, **kwargs)
    return ensure_future(wrap())


def future(func: Callable) -> Callable:
    """
    Let coroutine return future object.
    It can be used as decorator.
    """
    @wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> Future:
        return ensure_future(func(*args, **kwargs))
    return wrap


def run_async(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Assemble coroutine and run.

    For example...

    from chainiter import future, run_async

    @future
    async def hoge():
        return 'fuga'
    fuga = run_async(hoge())
    """
    result = func(*args, **kwargs)
    loop = new_event_loop()
    result = loop.run_until_complete(result)
    loop.close()
    return result

def as_sync(func):
    """
    Make an async function a normal function.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        loop = new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrap


def as_sync(func):
    """
    Make an async function a normal function.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        loop = new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrap


def make_color(txt: str, num: int) -> str:
    color = '\033[9' + str(6 - num) + 'm'
    reset = '\033[0m'
    return color + txt + reset


class ChainBase:
    def __init__(self, data: Union[Iterable],
                 indexable: bool = False):
        """
        Parameters
        ----------
        data: Iterable
            It need not to be indexable.
        indexable: bool
            If data is indexable, indexable should be True.
        """
        if not hasattr(data, '__iter__'):
            TypeError('It is not iterator')
        self.data = data
        self.indexable = indexable
        self._num = 0  # Iterator needs number.
        self.mode = mode.SINGLE

    def calc(self) -> 'ChainIter':
        """
        ChainIter.data may be list, map, filter and so on.
        This method translate it to list.

        Returns
        ----------
        ChainIter object with result.
        """
        self.data = list(self.data)
        return cast(ChainIter, self)


class ChainOperator(ChainBase):
    def __add__(self, item: Any) -> 'ChainIter':
        if not hasattr(self.data, 'append'):
            raise IndexError('Run ChainIter.calc().')
        cast(list, self.data).append(item)
        return cast(ChainIter, self)

    def __setitem__(self, key: Any, item: Any) -> None:
        if hasattr(self.data, '__setitem__'):
            cast(list, self.data)[key] = item
        raise IndexError('Item cannot be set. Run ChainIter.calc().')

    def __lt__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x < arg, self.data)).get()

    def __le__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x <= arg, self.data)).get()

    def __gt__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x > arg, self.data)).get()

    def __ge__(self, arg: Any) -> list:
        return ChainIter(map(lambda x: x >= arg, self.data)).get()


class ChainPrivate(ChainOperator):

    def __reversed__(self) -> Iterable:
        if hasattr(self.data, '__reversed__'):
            return cast(list, self.data).__reversed__()
        raise IndexError('Not reversible')

    def __len__(self) -> int:
        if not self.indexable:
            self.calc()
            self.indexable = True
        return len(cast(list, self.data))

    def __repr__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def __str__(self) -> str:
        return 'ChainIter[{}]'.format(str(self.data))

    def __getitem__(self, num: int) -> Any:
        if self.indexable or hasattr(self.data, '__getitem__'):
            return cast(list, self.data)[num]
        self.data = tuple(self.data)
        return self.data[num]

    def __iter__(self) -> 'ChainIter':
        self.calc()
        self._max = len(cast(list, self.data))
        self._num = 0
        self.start_time = self.current_time = time.time()
        # return self.data
        return cast(ChainIter, self)

    def __next__(self) -> Any:
        if self._num == self._max:
            self._num = 0
            raise StopIteration
        result = self.__getitem__(self._num)
        self._num += 1
        return result


class ChainIterBase(ChainPrivate):
    def append(self, item: Any) -> 'ChainIter':
        if not isinstance(self.data, list):
            self.calc()
        cast(list, self.data).append(item)
        return cast(ChainIter, self)

    def get(self, kind: type = list) -> Any:
        """
        Get data as list.

        Parameters
        ----------
        kind: Callable
            If you want to convert to object which is not list,
            you can set it. For example, tuple, dqueue, and so on.
        """
        return kind(self.data)


def _run_with_arg(arg: Tuple[Callable, tuple, dict]):
    return arg[0](*arg[1], **arg[2])


def _run_with_arg_async(arg: Tuple[Callable, tuple, dict]):
    result = arg[0](*arg[1], **arg[2])
    loop = new_event_loop()
    result = loop.run_until_complete(result)
    loop.close()
    return result


class ChainIterNormal(ChainIterBase):

    def zip(self, *args: Iterable) -> 'ChainIter':
        """
        Simple chainable zip function.

        Parameters
        ----------
        *args: Iterators to zip.

        Returns
        ----------
        Result of func(*ChainIter, *args, **kwargs)
        """
        return ChainIter(zip(self.data, *args), False)

    def reduce(self, func: Callable, logger: Logger = logger) -> Any:
        """
        Simple reduce function.

        Parameters
        ----------
        func: Callable
        logger: logging.Logger
            Your favorite logger.

        Returns
        ----------
        Result of reduce.
        """
        write_info(func, 1, logger)
        return reduce(func, self.data)

    def map(self, func: Callable, process: int = 1,
            thread: int = 1,
            args: tuple = (), kwargs: dict = {},
            timeout: Optional[float] = None,
            logger: Logger = logger) -> 'ChainIter':
        """
        Chainable map.

        Parameters
        ----------
        func: Callable
            Function to run.
        process: int
            Number of processes. Multiprocessing is a little heavy task.
            But it can use all of cpu cores.
            If it is not 1, multi threading will not be done.
        threads: int
            Number of threads. It can not use all of cpu cores because of GIL.
            But it may be faster if IO is big or you are using code
            which avoids GIL like numpy.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result

        >>> ChainIter([5, 6]).map(lambda x: x * 2).get()
        [10, 12]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            if args or kwargs:
                return ChainIter(map(
                        _run_with_arg,
                        ((func, (data,) + args, kwargs) for data in self.data)
                ), False)
            return ChainIter(map(func, self.data), False)
        elif process != 1:
            self.mode = mode.PROCESS
            write_info(func, process, logger, 'process')
            # _run_with_arg
            with Pool(process) as pool:
                result = pool.map_async(
                    _run_with_arg,
                    ((func, (data,) + args, kwargs) for data in self.data)
                ).get(timeout)
            return ChainIter(result, True)
        else:
            self.mode = mode.THREAD
            write_info(func, thread, logger, 'thread')
            with ThreadPoolExecutor(thread) as exe:
                result = exe.map(
                    _run_with_arg,
                    ((func, (data,) + args, kwargs) for data in self.data)
                )
            return ChainIter(result, True)

    def starmap(self, func: Callable, process: int = 1, thread: int = 1,
                args: tuple = (), kwargs: dict = {},
                timeout: Optional[float] = None,
                logger: Logger = logger) -> 'ChainIter':
        """
        Chainable starmap.
        In this case, ChainIter.data must be Iterator of iterable objects.

        Parameters
        ----------
        func: Callable
            Function to run.
        process: int
            Number of processes. Multiprocessing is a little heavy task.
            But it can use all of cpu cores.
            If it is not 1, multi threading will not be done.
        threads: int
            Number of threads. It can not use all of cpu cores because of GIL.
            But it may be faster if IO is big or you are using code
            which avoids GIL like numpy.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> def multest2(x, y): return x * y
        >>> ChainIter([5, 6]).zip([2, 3]).starmap(multest2).get()
        [10, 18]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(map(
                _run_with_arg,
                ((func, data + args, kwargs) for data in self.data)
            ), False)
            # return ChainIter(starmap(func, self.data),
            #                  False)
        elif process != 1:
            write_info(func, process, logger, 'single')
            with Pool(process) as pool:
                result = pool.map_async(
                    _run_with_arg,
                    ((func, data + args, kwargs) for data in self.data)).get(timeout)
            # with Pool(process) as pool:
            #     result = pool.starmap_async(func, self.data).get(timeout)
            return ChainIter(result, True)
        else:
            write_info(func, thread, logger, 'single')
            result = thread_starmap(func, thread, list(self.data))
            return ChainIter(result, True)

    def filter(self, func: Callable, logger: Logger = logger) -> 'ChainIter':
        """
        Simple filter function.

        Parameters
        ----------
        func: Callable
        logger: logging.Logger
            Your favorite logger.
        """
        write_info(func, 1, logger)
        return ChainIter(filter(func, self.data), False)




class ChainIterAsync(ChainIterBase):
    def async_map(self, func: Callable, process: int = 1, thread: int = 1,
                  args: tuple = (), kwargs: dict = {},
                  timeout: Optional[float] = None,
                  logger: Logger = logger) -> 'ChainIter':
        """
        Chainable map of coroutine, for example, async def function.

        Parameters
        ----------
        func: Callable
            Function to run.
        process: int
            Number of processes. Multiprocessing is a little heavy task.
            But it can use all of cpu cores.
            If it is not 1, multi threading will not be done.
        threads: int
            Number of threads. It can not use all of cpu cores because of GIL.
            But it may be faster if IO is big or you are using code
            which avoids GIL like numpy.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> async def multest(x): return x * 2
        >>> ChainIter([1, 2]).async_map(multest).get()
        [2, 4]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(
                map(_run_with_arg_async,
                    ((func, (data,) + args, kwargs) for data in self.data)),
                False)
        elif process != 1:
            write_info(func, process, logger, 'process')
            with Pool(process) as pool:
                result = pool.map_async(
                    _run_with_arg_async,
                    ((func, (data,) + args, kwargs) for data in self.data)
                ).get(timeout)
            return ChainIter(result, True)
        else:
            write_info(func, thread, logger, 'thread')
            with ThreadPoolExecutor(thread) as exe:
                result = exe.map(
                    _run_with_arg_async,
                    ((func, (data,) + args, kwargs) for data in self.data))
            return ChainIter(result, True)

    def async_starmap(self, func: Callable, process: int = 1, thread: int = 1,
                      timeout: Optional[float] = None,
                      logger: Logger = logger) -> 'ChainIter':
        """
        Chainable starmap of coroutine, for example, async def function.

        Parameters
        ----------
        func: Callable
            Function to run.
        process: int
            Number of processes. Multiprocessing is a little heavy task.
            But it can use all of cpu cores.
            If it is not 1, multi threading will not be done.
        threads: int
            Number of threads. It can not use all of cpu cores because of GIL.
            But it may be faster if IO is big or you are using code
            which avoids GIL like numpy.
        timeout: Optional[float] = None
            Time to stop parallel computing.
        logger: logging.Logger
            Your favorite logger.
        Returns
        ---------
        ChainIter with result
        >>> async def multest(x, y): return x * y
        >>> ChainIter(zip([5, 6], [1, 3])).async_starmap(multest).get()
        [5, 18]
        """
        if process == 1 and thread == 1:
            write_info(func, 1, logger, 'single')
            return ChainIter(
                starmap(run_async, ((func, *a) for a in self.data)),
                False)
        elif process != 1:
            write_info(func, process, logger, 'single')
            with Pool(process) as pool:
                result = pool.starmap_async(
                    run_async, ((func, *a) for a in self.data)).get(timeout)
            return ChainIter(result, True)
        else:
            write_info(func, thread, logger, 'single')
            result = thread_starmap(
                run_async, thread, ((func, *a) for a in self.data))
            return ChainIter(result, True)


class ChainMisc(ChainBase):
    def print(self) -> 'ChainIter':
        """
        Just print the content.
        """
        print(self.data)
        return cast(ChainIter, self)

    def log(self, logger: Logger = logger, level: int = INFO) -> 'ChainIter':
        """
        Just print the content.
        """
        if level == INFO:
            logger.info(self.data)
        elif level == WARNING:
            logger.warning(self.data)
        elif level == ERROR:
            logger.error(self.data)
        elif level == CRITICAL:
            logger.critical(self.data)
        return cast(ChainIter, self)

    def print_len(self) -> 'ChainIter':
        """
        Just print length of the content.
        """
        print(len(list(self.data)))
        return cast(ChainIter, self)

    def arg(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Use ChainIter object as argument.
        It is same as func(*ChainIter, *args, **kwargs)

        Parameters
        ----------
        func: Callable

        Returns
        ----------
        Result of func(*ChainIter, *args, **kwargs)
        >>> ChainIter([5, 6]).arg(sum)
        11
        """
        return func(tuple(self.data), *args, **kwargs)

    def stararg(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Use ChainIter object as argument.
        It is same as func(*tuple(ChainIter), *args, **kwargs)

        Parameters
        ----------
        func: Callable

        Returns
        ----------
        ChainIter object
        >>> ChainIter([5, 6]).stararg(lambda x, y: x * y)
        30
        """
        return func(*tuple(self.data), *args, **kwargs)


class ChainIter(ChainIterNormal, ChainMisc,
                ChainIterAsync):
    """
    Iterator which can used by method chain like Arry of node.js.
    Multi processing and asyncio can run.
    """
    def __init__(self, data: Union[list, Iterable],
                 indexable: bool = False):
        super(ChainIter, self).__init__(
            data, indexable)


def _tmp_thread(args: Any) -> Any: return args[0](*args[1:])
def thread_starmap(func: Callable, chunk: int = 1,
                   args: Union[list, Iterable] = (None,)) -> Iterable:
    with ThreadPoolExecutor(chunk) as exe:
        result = exe.map(_tmp_thread, ((func, *a) for a in args))
    return result

from itertools import repeat
from typing import List, Tuple, Dict, Optional
from subprocess import Popen, CalledProcessError, PIPE, STDOUT


def __apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def starstarmap(
        pool,
        fn, 
        args_iter: Optional[List] = None, 
        kwargs_iter: Optional[List] = None,
        total: Optional[int] = None,
    ):
    """pool.starmap that supports and iterator of keyword arguments

    Args:
        pool (_type_): _description_
        fn (function): _description_
        args_iter (List, optional): _description_. Defaults to None.
        kwargs_iter (List, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if args_iter is None and kwargs_iter is None:
        raise ValueError("args_iter and kwargs_iter cannot both be None.")
    
    if args_iter is None:
        args_iter = repeat([])

    if kwargs_iter is None:
        kwargs_iter  = repeat(dict())

    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)

    return pool.starmap_async(__apply_args_and_kwargs, args_for_starmap)

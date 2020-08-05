import json
import multiprocessing as mp
import os
import signal
from datetime import datetime, timedelta

import dateutil.parser
from tqdm import tqdm


def datetime_parser(json_dict):
    for (key, value) in json_dict.items():
        try:
            json_dict[key] = convert_to_timedelta(value)
        except (TypeError, ValueError, AttributeError):
            pass
    return json_dict


def convert_to_timedelta(value):
    return dateutil.parser.parse(value) - datetime.today().replace(hour=0, minute=0, second=0,
                                                                   microsecond=0)


def timeout(time_s, func, **kwargs):
    def raise_timeout(signum, frame):
        raise TimeoutError

    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time_s)

    try:
        res = func(**kwargs)
    except TimeoutError:
        print("timeout appends after {}s for func:{} with args:{}".format(time_s, func, kwargs))
        res = None
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return res


def pool_map_bar(func, args, n_processes=2):
    p = mp.Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def save_dict(dict, path, filename=None):
    write_path = os.path.join(path, filename) if filename else path
    open(write_path, 'w+').write(
        json.dumps(dict, indent=4, default=str)
    )


def get_json(json_file):
    return json.loads(open(json_file, 'r').read(), object_hook=datetime_parser)


def toc(tic=None):
    if tic is None:
        return datetime.now()
    return datetime.now() - tic


def myconverter(o):
    if isinstance(o, datetime) or isinstance(o, timedelta):
        return o.__str__()


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def normalize(data):
    return [(i - min(data)) / (max(data) - min(data)) for i in data] \
        if min(data) != max(data) \
        else [min(data) for _ in data]


def assert_sorted(list):
    assert all(list[i] <= list[i + 1] for i in range(len(list) - 1))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
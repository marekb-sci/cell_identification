# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import json
import pickle
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def save_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)

def save_json(obj, fn):
    with open(fn, 'w') as f:
        json.dump(obj, f)

def save_yaml(obj, fn):
    with open(fn, 'w') as f:
        yaml.dump(obj, f, Dumper=Dumper)

def load_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_json(fn):
    with open(fn) as f:
        obj = json.load(f)
    return obj

def load_yaml(fn):
    with open(fn) as f:
        obj = yaml.load(f, Loader=Loader)
    return obj

#%%
def parse_mode(mode, fn):
    if mode is None:
        fn = str(fn)
        if fn.endswith('.pickle') or fn.endswith('.pkl'):
            mode = 'pickle'
        elif fn.endswith('.json'):
            mode = 'json'
        elif fn.endswith('.yaml'):
            mode = 'yaml'
    if not mode in ['json', 'pickle', 'yaml']:
        raise ValueError(f'serialization mode unknown for {fn}, {mode}')
    return mode


def save(obj, fn, mode=None):
    mode = parse_mode(mode, fn)
    if mode == 'pickle':
        save_pickle(obj, fn)
    elif mode == 'json':
        save_json(obj, fn)
    elif mode == 'yaml':
        save_yaml(obj, fn)

def load(fn, mode=None):
    mode = parse_mode(mode, fn)
    if mode == 'pickle':
        return load_pickle(fn)
    elif mode == 'json':
        return load_json(fn)
    elif mode == 'yaml':
        return load_yaml(fn)
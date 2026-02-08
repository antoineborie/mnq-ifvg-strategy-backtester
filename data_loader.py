import pickle
import pandas as pd
import numpy as np
import os
import glob as glob_mod


class PlaceholderStringArray:
    _is_placeholder = True

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) >= 2:
            self._data = np.asarray(state[1], dtype=object)
        else:
            self._data = np.array([], dtype=object)

    @property
    def dtype(self):
        return np.dtype('O')

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)


class CompatStringDtype:
    def __new__(cls, storage='python', na_value=None):
        return pd.StringDtype(storage)


def _fake_pyx_unpickle(__pyx_type, __pyx_checksum, __pyx_state):
    if __pyx_type.__name__ == 'StringArray':
        return PlaceholderStringArray()
    from pandas._libs.arrays import __pyx_unpickle_NDArrayBacked
    return __pyx_unpickle_NDArrayBacked(__pyx_type, __pyx_checksum, __pyx_state)


import pandas._libs.internals as _pli
_original_unpickle_block = _pli._unpickle_block


def _patched_unpickle_block(values, placement, ndim):
    if isinstance(values, PlaceholderStringArray):
        arr = values._data
        if ndim == 2 and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _original_unpickle_block(arr, placement, ndim)
    return _original_unpickle_block(values, placement, ndim)


class FixedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'StringDtype':
            return CompatStringDtype
        if name == '__pyx_unpickle_NDArrayBacked':
            return _fake_pyx_unpickle
        if module == 'pandas._libs.internals' and name == '_unpickle_block':
            return _patched_unpickle_block
        return super().find_class(module, name)


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        df = FixedUnpickler(f).load()
    return df


def list_data_files(data_dir='data'):
    files = sorted(glob_mod.glob(os.path.join(data_dir, 'MNQ_*.pkl')))
    return files


def load_data(files):
    frames = []
    for f in files:
        df = load_pkl(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined = combined.sort_index()
    if combined.index.tz is None:
        combined = combined.tz_localize('UTC')
    return combined


def get_active_contract(df):
    if 'symbol' not in df.columns:
        return df
    df = df.copy()
    df['_date'] = df.index.date
    daily_vol = df.groupby(['_date', 'symbol'])['volume'].sum().reset_index()
    daily_vol.columns = ['date', 'symbol', 'volume']
    main_per_day = daily_vol.loc[daily_vol.groupby('date')['volume'].idxmax()]
    active_map = dict(zip(main_per_day['date'], main_per_day['symbol']))
    df['_active'] = df['_date'].map(active_map)
    result = df[df['symbol'] == df['_active']].drop(columns=['_date', '_active'])
    return result.sort_index()

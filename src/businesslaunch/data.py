"""Streaming readers for the Yelp JSON-lines files (plain .json or .json.zip)."""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd


def locate(kind, data_dir):
    """Path of yelp_academic_dataset_<kind>.json or its .zip inside data_dir."""
    for name in [f'yelp_academic_dataset_{kind}.json', f'yelp_academic_dataset_{kind}.json.zip', f'{kind}.json']:
        path = Path(data_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'Put yelp_academic_dataset_{kind}.json(.zip) in {data_dir} (see data/README.md).')


def iter_json_lines(path, kind):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('.json') and kind in Path(n).name]
            if len(names) != 1:
                raise ValueError(f'Expected one {kind} JSON inside {path}, found {names}')
            with archive.open(names[0]) as raw:
                for line in io.TextIOWrapper(raw, encoding='utf-8'):
                    if line.strip():
                        yield json.loads(line)
    else:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def read_philadelphia_businesses(path):
    rows = [r for r in iter_json_lines(path, 'business')
            if str(r.get('city', '')).strip().casefold() == 'philadelphia'
            and str(r.get('state', '')).strip().upper() == 'PA']
    return pd.DataFrame(rows)


def read_checkins(path, business_ids):
    """checkin_count, first_checkin, last_checkin for the given business ids."""
    keep, rows = set(business_ids), {}
    for r in iter_json_lines(path, 'checkin'):
        if r.get('business_id') in keep:
            dates = sorted(d.strip() for d in str(r.get('date', '')).split(',') if d.strip())
            if dates:
                rows[r['business_id']] = (len(dates), dates[0], dates[-1])
    frame = pd.DataFrame.from_dict(rows, orient='index', columns=['checkin_count', 'first_checkin', 'last_checkin'])
    frame['first_checkin'] = pd.to_datetime(frame['first_checkin'])
    frame['last_checkin'] = pd.to_datetime(frame['last_checkin'])
    return frame


def parse_price(attributes):
    if not isinstance(attributes, dict):
        return float('nan')
    try:
        value = float(str(attributes.get('RestaurantsPriceRange2')).strip("'\""))
    except (TypeError, ValueError):
        return float('nan')
    return value if value in {1, 2, 3, 4} else float('nan')

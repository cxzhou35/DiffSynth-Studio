import os
import csv
import pandas
import ujson as json
from argparse import Namespace
from os.path import join, exists, basename

class FileHandler():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.sp_file_types = ['csv', 'json']
        file_type = basename(file_path).split('.')[-1]
        assert file_type in self.sp_file_types, f"Unsupported file type: {file_type}"
        self.file_type = file_type
        self.data_container = self.create_data_container(file_type)

    def create_data_container(self, file_type: str):
        if file_type == 'csv':
            data_container = []
        elif file_type == 'json':
            data_container = DotDict()

        return data_container

    def update_data_container(self, data):
        if self.file_type == 'csv':
            self.data_container.append(data)
        elif self.file_type == 'json':
            self.data_container.update(data)

    def read(self):
        if self.file_type == 'csv':
            contents = self._read_csv(self.file_path)
        elif self.file_type == 'json':
            contents = self._read_json(self.self_path)

        return contents

    def write(self, data):
        if self.file_type == 'csv':
            self._write_csv(self.file_path, data)
        elif self.file_type == 'json':
            self._write_json(self.file_path, data)

    def _read_csv(self, file_path: str):
        data = []
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data

    def _write_csv(self, file_path: str, data_container):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for data in data_container:
                writer.writerow(data)

    def _read_json(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def _write_json(self, file_path: str, data_container):
        with open(file_path, 'w') as f:
            json.dump(data_container, f, indent=4, escape_forward_slashes=False)


class DotDict(dict):
    def __init__(self, mapping=None, /, **kwargs):
        if mapping is None:
            mapping = {}
        elif type(mapping) is Namespace:
            mapping = vars(mapping)

        super().__init__(mapping, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if type(value) is dict:
                value = DotDict(value)
            return value
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DotDict " + dict.__repr__(self) + ">"

    def todict(self):
        return {k: v for k, v in self.items()}


class default_dotdict(DotDict):
    def __init__(self, default_type=object, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        dict.__setattr__(self, "default_type", default_type)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (AttributeError, KeyError) as e:
            super().__setitem__(key, dict.__getattribute__(self, "default_type")())
            return super().__getitem__(key)

dotdict = DotDict

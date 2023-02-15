from typing import Dict, List

import pyarrow as pa
from tqdm import tqdm

from load_files_from_s3 import s3_listdir, smart_open
from joblib.memory import Memory
from pyarrow import  parquet as pq
from collections import OrderedDict

mem = Memory(".cache")


@mem.cache
def _obtain_length(file: str) -> int:
    table = pq.read_table(file)
    return len(table)


@mem.cache
def _obtain_all_lengths(files: List[str]) -> Dict[str, int]:
    length_map = OrderedDict()
    for file in tqdm(files, "extracting length of individual files"):
        length_map[file] = _obtain_length(file)
    return length_map


@mem.cache
def _indexing_files(length_map: Dict[str, int]):
    index_map = OrderedDict()
    current_index = 0
    for file, length in tqdm(length_map.items(), 'building indices'):
        index_map[file] = (current_index, length-1)
        current_index += length
    return index_map


class SharededParquetS3Dataset:

    def __init__(self, s3_url: str, hash_idx: int = 0, uri_index: int = 1, text_idx: int = 2):
        files = s3_listdir(s3_url, ".parquet")
        # ensure the files are allways the same
        files.sort()
        self.length_map = _obtain_all_lengths(files=files)
        self.index_map = _indexing_files(self.length_map)
        self.cache = None
        self.cache_name = None
        self.hash_index = hash_idx
        self.uri_index = uri_index
        self.text_idx = text_idx

    def _check_for_index_in_infile(self, idx: int, file: str) -> bool:
        return idx >= self.index_map[file][0] and idx < self.index_map[file][1]

    def _find_correct_file(self, idx: int) -> str:
        if self.cache_name is not None and self._check_for_index_in_infile(idx, self.cache_name):
            return self.cache_name
        for file in self.index_map.keys():
            if self._check_for_index_in_infile(idx, file):
                return file
        raise ValueError(f"index: {idx}, cannot find file in dataset of total length {self.__len__()}")

    def __len__(self):
        return sum(list(self.length_map.values()))

    def _obtain_true_index(self, idx: int, file: str):
        true_index = idx - self.index_map[file][0]
        assert true_index > 0
        return true_index

    def _obtain_with_true_index(self, true_index: int, file: str):
        # check if cache miss; load if necessary
        if file != self.cache_name:
            table = pq.read_table(file)
            self.cache = table
            self.cache_name = file

        return {
            "hash": self.cache[self.hash_index][true_index].as_py(),
            "uri": self.cache[self.uri_index][true_index].as_py(),
            "text": self.cache[self.text_idx][true_index].as_py()
        }

    def __getitem__(self, idx):
        file = self._find_correct_file(idx)
        true_index = self._obtain_true_index(idx=idx, file=file)
        return self._obtain_with_true_index(true_index=true_index, file=file)

if __name__ == '__main__':

    SharededParquetS3Dataset(s3_url="s3://s-laion/bild_text/run1/2023-02-07-23-32-48")

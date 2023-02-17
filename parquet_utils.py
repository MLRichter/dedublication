import time
from typing import Dict, List, Union

import pyarrow as pa
from filelock import FileLock
from tqdm import tqdm

from load_files_from_s3 import s3_listdir, smart_open
from joblib.memory import Memory
from pyarrow import  parquet as pq
from collections import OrderedDict
import os
import boto3

mem = Memory(".cache")


def _obtain_length(file: str) -> int:
    meta = pq.read_metadata(file)
    return meta.num_rows


def fetch(bucket_name: str, key: str, abs_path: str, s3_client):
    file = f'{abs_path}/{key}'
    os.makedirs(file, exist_ok=True)
    with open(file, 'wb') as data:
        s3_client.download_fileobj(bucket_name, key, data)
    return file


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
        index_map[file] = (current_index, current_index+length-1)
        current_index += length
    return index_map


class SharededParquetS3Dataset:

    def __init__(self, s3_url: Union[str, List[str]], hash_idx: int = 0, uri_index: int = 1, text_idx: int = 2, lock: str = "shard.lock", batch_size: int = 10000, timeout: int = 6000, slurmprocess_filelock: bool = True, slurm_process_per_lock: int = 16):
        if "SLURM_PROCID" in os.environ and slurmprocess_filelock:
            lock = "shard{}.lock".format(int(os.environ["SLURM_PROCID"]) % slurm_process_per_lock)

        files = s3_listdir(s3_url, ".parquet") if isinstance(s3_url, str) else s3_url
        # ensure the files are allways the same
        files.sort()
        self.lock = FileLock(lock_file=lock, timeout=timeout)
        index_lock = FileLock(lock_file="index.lock", timeout=300)
        with index_lock:
            self.length_map = _obtain_all_lengths(files=files)
            self.index_map = _indexing_files(self.length_map)
        self.cache = None
        self.cache_name = None
        self.cache_ed_indices = None
        self.hash_index = hash_idx
        self.uri_index = uri_index
        self.text_idx = text_idx
        self.s3 = boto3.client('s3')
        self.batch_size = batch_size
        self.index_range = None

    def _check_for_index_in_infile(self, idx: int, file: str) -> bool:
        return idx >= self.index_map[file][0] and idx <= self.index_map[file][1]

    def _find_correct_file(self, idx: int) -> str:
        if self.cache_name is not None and self._check_for_index_in_infile(idx, self.cache_name):
            return self.cache_name
        for file in self.index_map.keys():
            if self._check_for_index_in_infile(idx, file):
                return file
        raise ValueError(f"index: {idx}, cannot find file in dataset of total length {self.__len__()}, the index map being: {self.index_map}")

    def __len__(self):
        return sum(list(self.length_map.values()))

    def _obtain_true_index(self, idx: int, file: str):
        true_index = idx - self.index_map[file][0]
        assert true_index >= 0
        return true_index

    def _check_index_cache_hit(self, true_index: int):
        if self.index_range is None:
            return True
        else:
            return not (true_index >= self.index_range[0] and true_index < self.index_range[1])

    def _obtain_with_true_index(self, true_index: int, file: str):
        # check if cache miss; load if necessary
        if file != self.cache_name or self._check_index_cache_hit(true_index=true_index):
            self.index_range = (
                true_index,
                min(
                    true_index + self.batch_size,
                    self.index_map[file][1]+1
                )
            )
            with self.lock.acquire():
                table = pq.read_table(file)[self.index_range[0]:self.index_range[1]]
            self.cache = table
            self.cache_name = file

        # batch index is the true index relative to the start of the batch window
        batch_index = true_index - self.index_range[0]
        assert true_index >= 0

        data_point =  {
            "hash": self.cache[self.hash_index][batch_index].as_py(),
            "uri": self.cache[self.uri_index][batch_index].as_py(),
            "text": self.cache[self.text_idx][batch_index].as_py()
        }
        return data_point

    def __getitem__(self, idx):
        file = self._find_correct_file(idx)
        true_index = self._obtain_true_index(idx=idx, file=file)
        return self._obtain_with_true_index(true_index=true_index, file=file)


if __name__ == '__main__':

    def billions_parquet_dataset():
        all_files = []
        for part in range(5):
            folder = "s3://s-laion/bild_text/run1/2023-02-07-23-32-48/part_{}/".format(part)
            files = s3_listdir(folder, ".parquet")
            all_files.extend(files)
        return SharededParquetS3Dataset(all_files, batch_size=50000)

    ds = billions_parquet_dataset()
    print("Found", len(ds), "datapoints")
    for i in range(0, 12000):
        start = time.time()
        datapoint = ds[i]
        total = time.time() - start
        print("took", total, "seconds")
        if i == 0:
            print("Sleeping")
            time.sleep(10)
    print("sample datapoint", ds[42])
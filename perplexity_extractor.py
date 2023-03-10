import datetime
from pathlib import Path
from time import sleep
from typing import List

import click
import datasets
import tqdm
from datasets.packaged_modules.pandas import pandas
from joblib import Parallel, delayed

from riverbed.kenlm_manager import *
import pandas as pd
from parquet_utils import SharededParquetS3Dataset
from load_files_from_s3 import s3_listdir

DATASET = None
MODELS = None


def billions_parquet_dataset():
    all_files = []
    for part in range(5):
        folder = "s3://s-laion/bild_text/run1/2023-02-07-23-32-48/part_{}/".format(part)
        files = s3_listdir(folder, ".parquet")
        all_files.extend(files)
    return SharededParquetS3Dataset(all_files, batch_size=300000)


dataset_mapper = {"c4_15M": lambda: datasets.load_dataset('teven/c4_15M', "binary")["train"],
                  "parquet1": lambda: datasets.load_dataset("parquet", data_files="../test/*.parquet")["train"],
                  "parquet3b": lambda: billions_parquet_dataset(),
                  }

def obtain_dataset(sample_start: int = 0, sample_end: int = None, dataset_key = "c4_15M"):
    global DATASET
    print("Fetching dataset", dataset_key)
    if DATASET is None:
        ds = dataset_mapper[dataset_key]()
        DATASET = ds
    else:
        ds = DATASET
    return ds


def get_model_dict() -> dict:
    global MODELS
    if MODELS is not None:
        return MODELS
    kenlm_model = load_kenlm_model(pretrained_models=["ontocord/riverbed_kenlm"])
    kenlm_model2 = load_kenlm_model("en", pretrained_models=['ccnet/wikipedia'])
    kenlm_model.update(kenlm_model2)
    print("models loaded:", list(kenlm_model.keys()))
    return kenlm_model


def obtain_perplexity(models: dict, text: str) -> Dict[str, int]:
    perplexities = {}
    for model_name, model in models.items():
        perplexities[model_name] = model.get_perplexity(text)
    return perplexities


def split(length, chunk_size):
    indices = list(range(0, length, chunk_size))
    start_stop = [(start, stop) for start, stop in zip(indices[:-1], indices[1:])]
    last_stop = start_stop[-1][-1]
    if last_stop < length:
        start_stop.append((last_stop, length))
    return start_stop


def ds_size(n_samples: int = None, dataset_key: str = None):
    return len(obtain_dataset(sample_end=n_samples, dataset_key=dataset_key))


def add_perplexities_to_result(result: Dict[str, List[int]], perplexities: Dict[str, int]) -> Dict[str, List[int]]:
    for model, perplexity in perplexities.items():
        result[f"perpl_{model}"].append(perplexity)
    return result


def process_chunk(start: int, stop: int, index: int, ds_key: str):
    models = get_model_dict()
    result_csv = {f"perpl_{model}": [] for model in models.keys()}
    result_csv["idx"] = []
    dataset = obtain_dataset(0, dataset_key=ds_key)
    print("Loaded dataset chunk:", index)
    for idx in range(start, stop):
        try:
            text = dataset[idx]["text"]
        except (IndexError, ValueError):
            print("Could not load", idx)
            continue
        perplexities = obtain_perplexity(models, text)
        result_csv = add_perplexities_to_result(result_csv, perplexities)
        result_csv["idx"].append(idx)
    return result_csv


def process_chunks_in_sequence(chunks,
                               sv_file: str = "./data/chunk{}_to_{}_results.csv",
                               ds_key: str = None,
                               rank: int = 0):
    files = []
    for idx, (start, stop) in enumerate(tqdm.tqdm(chunks, desc=f"rank: {rank}")):
        job = (start, stop, idx, ds_key, sv_file.format(start, stop))
        file = do_process_chunk(job)
        files.append(file)
    return files


def do_process_chunk(args):
    if os.path.exists(args[-1]):
        print(args[-1], "found allready, assuming chunk is processed")
        return args[-1]
    process_args = args[:-1]
    print(process_args)
    csv = process_chunk(*process_args)
    pd.DataFrame.from_dict(csv).to_csv(args[-1])
    return args[-1]


def process_chunks_in_parallel(chunks,
                               n_jobs: int = 2,
                               sv_file: str = "./data/chunk{}_to_{}_results.csv",
                               ds_key: str = None):
    parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    jobs = []
    for idx, (start, stop) in enumerate(tqdm.tqdm(chunks)):
        job = (start, stop, idx, ds_key, sv_file.format(start, stop))
        jobs.append(job)
    files = parallel(delayed(do_process_chunk)(job) for job in tqdm.tqdm(jobs))
    return files

def unify(savefiles: List[str], template: str):
    csv_file = template.format("all", "unified")
    print("unifying the following files:")
    for f in savefiles:
        print("\t", f)
    if os.path.exists(csv_file):
        print("Detected previous unification attempts, removing...")
        os.remove(csv_file)
    for file in tqdm.tqdm(savefiles, "Unifying dataframes"):
        csv = pd.read_csv(file, index_col=0)
        if os.path.exists(csv_file):
            pd.DataFrame.from_dict(csv).to_csv(csv_file, mode='a', header=False)
        else:
            pd.DataFrame.from_dict(csv).to_csv(csv_file)


def wait_for_other_processes_to_finish(world_size: int, max_wait_periods = 60):
    def check_for_completion_files():
        waiting_for = []
        for i in range(world_size):
            if not os.path.exists(f"{i}.done"):
                print(f"waiting for rank {i} to finish")
                waiting_for.append(i)
        return waiting_for

    for _ in range(max_wait_periods):
        if len(check_for_completion_files()) == 0:
            print("All processes have finished")
            break
        else:
            sleep(60)


def wait_for_other_ranks_to_finish_if_necessary(rank: int, world_size: int):
    if rank != 0 or world_size == 1:
        print("Rank", rank, "World_Size", world_size, "therefore no waiting necessary")
        return
    else:
        wait_for_other_processes_to_finish(world_size)


def fetch_files(savefile_template: str, chunk_size: int, n_samples: int):
    chunks = split(n_samples, chunk_size)
    filenames = []
    for idx, (start, stop) in enumerate(tqdm.tqdm(chunks)):
        file = savefile_template.format(start, stop)
        filenames.append(file)
    return filenames


@click.command()
@click.option('--n_samples', default=1000000, help="number of samples to process")
@click.option('--chunk_size', default=10000, help="size of a chunk processed by one worker")
@click.option('--multiprocessing', default=2, help="number of processes, 0 is for sequential processing")
@click.option('--sv_file', default="./data/chunk{}_to_{}_results.csv", help="format string with two slots for start and stop-sample for the respective chunk. Must be a valid path to a .csv-file")
@click.option('--rank', default=0, help="rank of the process in a multi-node setup, this will be overwritten by SLURM_PROCID if this environment variable exists")
@click.option('--world_size', default=1, help="how often this script is executed in parallel")
@click.option('--unify_chunks', default=True, help="if chunking is enabled due to multiprocessing, the chunks will be unified in the main thread")
@click.option('--dataset', default="c4_15M", help="if chunking is enabled due to multiprocessing, the chunks will be unified in the main thread")
def main(n_samples: int = -1,
         chunk_size: int = 1500000,
         multiprocessing: int = 2,
         sv_file: str = "./data/chunk{}_to_{}_results.csv",
         rank: int = 0,
         world_size: int = 1,
         unify_chunks: bool = True,
         dataset: str = "c4_15M"):

    if rank != 0:
        rank = rank if os.environ["SLURM_PROCID"] is None else int(os.environ["SLURM_PROCID"])
    print(f"Rank {rank}: booting up")

    n_samples = n_samples if n_samples != -1 else ds_size(dataset_key=dataset) + 1
    chunks = split(n_samples, chunk_size)
    if world_size != 1:
        print("Executing script within a context of world size", world_size)
        n_chunks_per_rank = len(chunks) // world_size
        start_chunk_idx, stop_chunk_idx = rank*n_chunks_per_rank, (rank+1)*n_chunks_per_rank
        hangovers = len(chunks)%world_size
        newchunks = chunks[start_chunk_idx:stop_chunk_idx]
        if hangovers != 0 and rank < hangovers:
            lefover = chunks[-1*(rank+1)]
            newchunks.append(lefover)
        chunks = newchunks
    else:
        print("world size", world_size, "single node mode enabled")
    print("dataset has size:", n_samples)
    print("will be processed in", len(chunks), "chunks")
    print("The chunks are:")
    for i, chunk in enumerate(chunks):
        print(f"\t({i})", chunk[0], ":", chunk[1])

    if os.path.exists(f"{rank}.done"):
        os.remove(f"{rank}.done")
    if multiprocessing != 1:
        print("multiprocessing enabled")
        files = process_chunks_in_parallel(chunks, n_jobs=multiprocessing, sv_file=sv_file, ds_key=dataset)
    else:
        print("multiprocessing disabled")
        files = process_chunks_in_sequence(chunks, sv_file=sv_file, ds_key=dataset, rank=rank)

    if unify_chunks:
        with open(f"{rank}.done", "w"):
            pass
        wait_for_other_ranks_to_finish_if_necessary(rank=rank, world_size=world_size)

        if rank == 0:
            if world_size != 1:
                print("Fetching all files")
                files = fetch_files(sv_file, chunk_size, n_samples)
            unify(savefiles=files, template=sv_file)


if __name__ == '__main__':
    main()


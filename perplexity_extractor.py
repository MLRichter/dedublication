from typing import List

import click
import datasets
import tqdm
from datasets.packaged_modules.pandas import pandas
from joblib import Parallel, delayed

from riverbed.kenlm_manager import *
import pandas as pd

DATASET = None
MODELS = None

def obtain_dataset(sample_start: int = 0, sample_end: int = None):
    global DATASET
    if DATASET is None:
        ds = datasets.load_dataset('teven/c4_15M', "binary")["train"]
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
    return start_stop


def ds_size(n_samples: int = None):
    return len(obtain_dataset(sample_end=n_samples))


def add_perplexities_to_result(result: Dict[str, List[int]], perplexities: Dict[str, int]) -> Dict[str, List[int]]:
    for model, perplexity in perplexities.items():
        result[f"perpl_{model}"].append(perplexity)
    return result


def process_chunk(start: int, stop: int, index: int):
    models = get_model_dict()
    result_csv = {f"perpl_{model}": [] for model in models.keys()}
    result_csv["idx"] = []
    dataset = obtain_dataset(0)
    print("Loaded dataset chunk:", index)
    for idx in range(start, stop):
        text = dataset[idx]["text"]
        perplexities = obtain_perplexity(models, text)
        result_csv = add_perplexities_to_result(result_csv, perplexities)
        result_csv["idx"].append(idx)
    return result_csv


def process_chunks_in_sequence(chunks, csv_file: str = "results.csv"):
    for idx, (start, stop) in enumerate(tqdm.tqdm(chunks)):
        csv = process_chunk(start, stop, idx)
        if os.path.exists(csv_file):
            pd.DataFrame.from_dict(csv).to_csv(csv_file, mode='a', header=False)
        else:
            pd.DataFrame.from_dict(csv).to_csv(csv_file)


def do_process_chunk(args):
    process_args = args[:-1]
    print(process_args)
    csv = process_chunk(*process_args)
    pd.DataFrame.from_dict(csv).to_csv(args[-1])


def process_chunks_in_parallel(chunks,
                               n_jobs: int = 2,
                               sv_file: str = "./data/chunk{}_to_{}_results.csv"):
    parallel = Parallel(n_jobs=n_jobs, backend="threading")
    jobs = []
    for idx, (start, stop) in enumerate(tqdm.tqdm(chunks)):
        job = (start, stop, idx, sv_file.format(start, stop))
        jobs.append(job)
    parallel(delayed(do_process_chunk)(job) for job in tqdm.tqdm(jobs))

@click.command()
@click.option('--n_samples', default=1000000, help="number of samples to process")
@click.option('--chunk_size', default=10000, help="size of a chunk processed by one worker")
@click.option('--multiprocessing', default=2, help="number of processes, 0 is for sequential processing")
@click.option('--sv_file', default="./data/chunk{}_to_{}_results.csv", help="format string with two slots for start and stop-sample for the respective chunk. Must be a valid path to a .csv-file if multiprocessing is enabled")
@click.option('--rank', default=0, help="rank of the process in a multi-node setup")
@click.option('--world_size', default=1, help="how often this script is executed in parallel")
def main(n_samples: int,
         chunk_size: int = 1500000,
         multiprocessing: int = 2,
         sv_file: str = "./data/chunk{}_to_{}_results.csv",
         rank: int = 0,
         world_size: int = 1):
    size = ds_size()
    chunks = split(n_samples, chunk_size)
    if world_size != 1:
        print("Executing script within a context of world size", world_size)
        n_chunks_per_rank = len(chunks) // world_size
        start_chunk_idx, stop_chunk_idx = rank*n_chunks_per_rank, (rank+1)*n_chunks_per_rank
        chunks = chunks[start_chunk_idx:stop_chunk_idx]
    else:
        print("world size", world_size, "single node mode enabled")
    print("dataset has size:", size)
    print("will be processed in", len(chunks), "chunks")
    print("The chunks are:")
    for i, chunk in enumerate(chunks):
        print(f"\t({i})", chunk[0], ":", chunk[1])

    if multiprocessing:
        print("multiprocessing enabled")
        process_chunks_in_parallel(chunks, n_jobs=multiprocessing, sv_file=sv_file)
    else:
        print("multiprocessing disabled")
        process_chunks_in_sequence(chunks)



if __name__ == '__main__':
    main()


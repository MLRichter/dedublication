import json
import os.path
import timeit
from typing import List

import click
import datasets
import pandas as pd
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed

from perplexity_extractor import split

DATASET = None
MODELS = None

dataset_mapper = {"c4_15M": lambda: datasets.load_dataset('teven/c4_15M', "binary")["train"],
                  "parquet1": lambda: datasets.load_dataset("parquet", data_files="../test/*.parquet")["train"]}

def obtain_dataset(sample_start: int = 0, sample_end: int = None, dataset_key = "c4_15M"):
    global DATASET
    print("Fetching dataset", dataset_key)
    if DATASET is None:
        ds = dataset_mapper[dataset_key]()
        DATASET = ds
    else:
        ds = DATASET
    return ds

def process():
    ...

import pandas as pd

def filter_dataframe(df, idx, epsilon=0.1):
    row = df.loc[idx]
    filtered_df = df[(df['perpl_ontocord/riverbed_kenlm'] >= row['perpl_ontocord/riverbed_kenlm'] - epsilon) & (df['perpl_ontocord/riverbed_kenlm'] <= row['perpl_ontocord/riverbed_kenlm'] + epsilon) &
                     (df['perpl_ccnet/wikipedia'] >= row['perpl_ccnet/wikipedia'] - epsilon) & (df['perpl_ccnet/wikipedia'] <= row['perpl_ccnet/wikipedia'] + epsilon)]

    return filtered_df


DF1 = None
DF2 = None


def filter_dataframe_optimized(df, idx, epsilon=0.1, df1=None, df2=None):


    window_size = 100

    ordinal_index1 = df1[df1.idx == idx].index.values[0]
    ordinal_index2 = df2[df2.idx == idx].index.values[0]

    start1 = max(0, ordinal_index1 - window_size)
    end1 = min(len(df1), ordinal_index1 + window_size)
    start2 = max(0, ordinal_index2 - window_size)
    end2 = min(len(df2), ordinal_index2 + window_size)

    filtered_df1 = df1.loc[start1:end1]
    filtered_df2 = df2.loc[start2:end2]

    if len(filtered_df1) == 0 or len(filtered_df2) == 0:
        return None

    filtered_df = filtered_df1[filtered_df1.idx.isin(filtered_df2.idx)]
    filtered_df = filtered_df.set_index("idx")
    if len(filtered_df) == 0:
        return filtered_df

    row = df.loc[idx]
    filtered_df = filtered_df[
        (filtered_df['perpl_ontocord/riverbed_kenlm'] >= row['perpl_ontocord/riverbed_kenlm'] - epsilon) & (filtered_df['perpl_ontocord/riverbed_kenlm'] <= row['perpl_ontocord/riverbed_kenlm'] + epsilon) &
        (filtered_df['perpl_ccnet/wikipedia'] >= row['perpl_ccnet/wikipedia'] - epsilon) & (filtered_df['perpl_ccnet/wikipedia'] <= row['perpl_ccnet/wikipedia'] + epsilon)]
    return filtered_df


def make_job(df, idx: int, data_folder = "./duplicates", df1=None, df2=None, dataset_name: str = "parquet1"):
    found = 0
    exact_match = 0
    near_match = 0
    epsilon = 0.05
    nearest_neighbours = filter_dataframe_optimized(df, idx, epsilon, df1, df2)
    #nn2 = filter_dataframe_optimized(df, idx, epsilon)
    #assert (nearest_neighbours.sort_index() == nn2.sort_index()).all().all()
    if nearest_neighbours is None:
        return []
    indices = nearest_neighbours.reset_index()["idx"].to_list()
    original = ""
    orig_idx = idx

    if len(indices) > 1:
        ds = obtain_dataset(dataset_key=dataset_name)
        real = ds[idx]["text"]
        original = ds[idx]["text"] + "\n" + ("=" * len(ds[idx]["text"]))
        for i, idx in enumerate(indices):
            duplicate = ds[idx]["text"]
            original = original + f"{i}. ({idx}) " + duplicate + ("\n"*3)
            if duplicate == real:
                if idx != orig_idx:
                    exact_match += 1
            else:
                near_match += 1
        # print(ds[5])
        # print(ds[227256])
        original += "=" * len(ds[idx]["text"]) + "\n"
        original += f"{len(indices)} +  duplicates found"
        #found += 1
        with open(os.path.join(data_folder, f"{idx}_found{found}_near{near_match}_exact{exact_match}.txt"), "w") as fp:
            fp.write(original)
        return indices


    return []


def do_job(args):
    return make_job(*args)

def process_chunk(idxs: List[int], df: pd.DataFrame, df1: pd.DataFrame, df2: pd. DataFrame, out_dir: str, dataset_name: str, rank: int):
    results = []
    for i, idx in enumerate(idxs):
        idx = int(idx)
        result = make_job(df, idx, out_dir, df1, df2, dataset_name)
        results.append(result)
        if i%10000 == 0:
            print("rank", rank, "processed", i, "samples")
    return results

def do_process_chunk(args):
    return process_chunk(*args)

@click.command()
@click.option('--n_jobs', default=8, help="number of processes")
@click.option('--csv_file', default="results.csv", help="source file containing perplexities")
@click.option('--out_dir', default="results.csv", help="source file containing the results")
@click.option('--dataset_name', default="parquet1", help="source file containing the results")
@click.option('--world_size', default=1, help="source file containing the results")
def main(n_jobs: int = 8, csv_file: str = "results.csv", out_dir: str = "./duplicates", dataset_name: str = "parquet1", world_size: int = 1):
    df = pd.read_csv(csv_file, index_col="idx")
    df1 = df.sort_values('perpl_ontocord/riverbed_kenlm').reset_index()
    df2 = df.sort_values('perpl_ccnet/wikipedia').reset_index()

    rank = int(os.environ["SLURM_PROCID"])

    parallel = Parallel(n_jobs=n_jobs)

    idxs: List[List[int]] = np.split(df.index.values, list(
        range(0, len(df), (len(df)//(n_jobs*world_size))))
                    )[1:]

    num_chunks = len(idxs) // world_size
    start = num_chunks * rank
    stop = num_chunks * (rank+1) if not (rank+1) == world_size else num_chunks
    print("rank", rank, "processes chunks with chunk index", start, "to", stop, "of total", len(idxs), "chunks processed by world size", world_size)
    idxs = idxs[start:stop]

    jobs = []
    for i, idx in enumerate(tqdm.tqdm(idxs, "Building Indices")):
        jobs.append((idx, df, df1, df2, out_dir, dataset_name, i))
    result = parallel(delayed(do_process_chunk)(x) for x in tqdm.tqdm(jobs, "Processing samples"))
    json_result = {int(idx): [int(d) for d in duplicates] for idx, duplicates in zip(df.index.to_list(), result)}
    with open(f"duplicates_{dataset_name}.json", "w") as fp:
        json.dump(json_result, fp)



if __name__ == '__main__':
    main()
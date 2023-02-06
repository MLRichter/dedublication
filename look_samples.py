import os.path
import timeit

import datasets
import pandas as pd
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed

DATASET = None
MODELS = None

def obtain_dataset(sample_start: int = 0, sample_end: int = None):
    global DATASET
    if DATASET is None:
        ds = datasets.load_dataset('teven/c4_15M', "binary", data_dir=r"./data/")["train"]
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


def make_job(df, idx: int, data_folder = "./duplicates", df1=None, df2=None):
    found = 0
    exact_match = 0
    near_match = 0
    epsilon = 0.05
    nearest_neighbours = filter_dataframe_optimized(df, idx, epsilon, df1, df2)
    #nn2 = filter_dataframe_optimized(df, idx, epsilon)
    #assert (nearest_neighbours.sort_index() == nn2.sort_index()).all().all()
    if nearest_neighbours is None:
        return (0, 0, 0)
    indices = nearest_neighbours.reset_index()["idx"].to_list()
    original = ""
    orig_idx = idx

    if len(indices) > 1:
        ds = obtain_dataset()
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
        found += 1
        with open(os.path.join(data_folder, f"{idx}_found{found}_near{near_match}_exact{exact_match}.txt"), "w") as fp:
            fp.write(original)

    return near_match, exact_match, found


def do_job(args):
    return make_job(*args)


def main():
    found = 0
    exact_match = 0
    near_match = 0

    df = pd.read_csv("results.csv", index_col="idx")
    df1 = df.sort_values('perpl_ontocord/riverbed_kenlm').reset_index()
    df2 = df.sort_values('perpl_ccnet/wikipedia').reset_index()
    parallel = Parallel(n_jobs=8)
    jobs = []
    for idx in tqdm.tqdm(list(range(1000000))):
        jobs.append((df, idx, "./duplicates", df1, df2))
    result = parallel(delayed(do_job)(x) for x in tqdm.tqdm(jobs))
    near_match, exact_match, found = tuple(sum(x) for x in zip(*result))
    """
        epsilon = 0.05
        nearest_neighbours = filter_dataframe(df, idx, epsilon)
        indices = nearest_neighbours.reset_index()["idx"].to_list()
        if indices is None:
            continue
        if len(indices) > 1:
            ds = obtain_dataset()
            original = ds[idx]["text"]
            print()
            print("="*len(ds[idx]["text"]))
            for i, idx in enumerate(indices):
                duplicate = ds[idx]["text"]
                print(f"{i}. ({idx})", duplicate)
                print()
                print()
                if duplicate == original:
                    exact_match += 1
                else:
                    near_match += 1
            #print(ds[5])
            #print(ds[227256])
            print("="*len(ds[idx]["text"]))

            print()
            print()
            print(len(indices), "duplicates found")
            print("="*len(ds[idx]["text"]))
            print("="*len(ds[idx]["text"]))
            print("="*len(ds[idx]["text"]))
            found += 1
    """


    print("Near Matches:\t", near_match)
    print("Exact. Matches:\t", exact_match)



if __name__ == '__main__':
    main()
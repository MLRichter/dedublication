import json
from typing import List, Tuple, Set

import click
from tqdm import tqdm

from utils import obtain_dataset
import os


def fetch_paths(path: str) -> List[str]:
    files = []
    for file in os.listdir(path):
        files.append(os.path.join(path, file))
    return files


def load_duplicate_info(path: str) -> Set[frozenset[int]]:
    paths = fetch_paths(path)
    all_duplicates = set()
    for file in tqdm(paths, "loading chunks"):
        with open(file, "r") as fp:
            duplicates_from_one_file = json.load(fp)
            for duplicates in duplicates_from_one_file:
                all_duplicates.add(frozenset(duplicates))
    return all_duplicates


def get_dataset_indices(dataset):
    return list(range(len(dataset)))


def keep_first(duplicate_instace: frozenset[int], all_to_keep: Set[int]):
    #if len(duplicate_instace) == 0:
    #    return duplicate_instace
    if len(duplicate_instace) != 0 and len(all_to_keep.intersection(duplicate_instace)) == 0:
        to_remove = set(duplicate_instace)
        to_keep = to_remove.pop()
        all_to_keep.add(to_keep)
        return all_to_keep
    else:
        return all_to_keep


def drop_all(duplicate_instace: frozenset[int], all_to_remove: Set[int]):
    return duplicate_instace


REMOVAL_STRATEGY = {
    "keep_first": keep_first,
    "drop_all": drop_all,
}


def remove_duplicates(duplicates: Set[Set[int]], dataset_indices: List[int], removal_strategy: str):

    processed_duplicates = [REMOVAL_STRATEGY[removal_strategy](duplicate) for duplicate in duplicates]
    all_indices_to_remove = set()
    for removal_chunk in processed_duplicates:
        all_indices_to_remove.union(removal_chunk)
    print(f"removing a total of {len(all_indices_to_remove)} from {len(dataset_indices)} ({round(len(all_indices_to_remove) / len(dataset_indices), 4)*100}%)")
    indices_to_keep = [idx for idx in dataset_indices if idx not in all_indices_to_remove]
    return indices_to_keep


def clean_dataset(dataset, path: str, removal_strategy: str) -> List[int]:
    duplicates = load_duplicate_info(path)
    dataset_indices = get_dataset_indices(dataset)
    indices_to_keep = remove_duplicates(duplicates=duplicates, dataset_indices=dataset_indices, removal_strategy=removal_strategy)
    cleaned_dataset = dataset.select(indices_to_keep)
    return cleaned_dataset


@click.command()
@click.option('--src_path', default="./duplicates/", help="folder containing duplicate info")
@click.option('--n_jobs', default=8, help="number of processes")
@click.option('--dataset_name', default="c4_15M", help="key for the dataset in question")
@click.option('--removal_strategy', default="keep_first", help="key specifying the removal strategy")
@click.option('--out_dir', default="./cleaned_dataset", help="output path for the cleaned dataset")
def main(src_path: str, n_jobs: int, dataset_name: str, removal_strategy: str, out_dir: str):

    # fetching dataset
    dataset = obtain_dataset(dataset_key=dataset_name)

    # creating the new dataset
    cleaned_dataset = clean_dataset(dataset=dataset, path=src_path, removal_strategy=removal_strategy)

    # saving the new dataset
    cleaned_dataset.save_to_disk(out_dir, num_proc=n_jobs)


if __name__ == '__main__':
    main()
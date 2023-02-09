import datasets


DATASET = None

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
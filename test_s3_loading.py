from datasets import load_dataset_builder

data_files = {"train": "bild_text/run1/2023-02-07-23-32-48/part_1/*.parquet"}
output_dir = "s3://s-laion/bild_text/run1/2023-02-07-23-32-48/"
builder = load_dataset_builder("parquet", data_files=data_files)
builder.download_and_prepare(output_dir, file_format="parquet")
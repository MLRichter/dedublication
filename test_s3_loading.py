from datasets import load_dataset_builder
import datasets
data_files = {"train": "s3://s-laion/bild_text/run1/2023-02-07-23-32-48/part_1/*.parquet"}
output_dir = "s3://s-laion/bild_text/run1/2023-02-07-23-32-48/"
builder = load_dataset_builder("parquet", data_files=data_files)
s3 = datasets.filesystems.S3FileSystem(anon=False)
builder.download_and_prepare(output_dir, storage_options=s3.storage_options, file_format="parquet")
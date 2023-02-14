from typing import Tuple, List, Optional

import boto3
import tqdm
from smart_open import smart_open


# example: s3://s-laion/bild_text/run1/2023-02-07-23-32-48/
def split_s3_path(path: str) -> Tuple[str, str]:
    bucket_with_prefix = path.split("://")[-1]
    bucket, prefix = bucket_with_prefix.split("/", maxsplit=1)
    return bucket, prefix


def list_all_files_with_prefix(bucket, prefix, filext=None):
    s3 = boto3.resource('s3')
    result = s3.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    filenames = []
    for cont in tqdm.tqdm(result["Contents"], "Reading files from S3"):
        filename = cont["Key"]
        if filext is not None and not filename.endswith(filext):
            continue
        filenames.append(filename)
    return filenames


def reassemble_uri(bucket: str, filename: str):
    return rf"s3://{bucket}/{filename}"


def reassemble_uris(bucket: str, filenames: List[str]):
    return [reassemble_uri(bucket, filename) for filename in filenames]


def s3_listdir(s3_path: str, filext: Optional[str] = None):
    bucket, prefix = split_s3_path(s3_path)
    filenames = list_all_files_with_prefix(bucket, prefix, filext=filext)
    return reassemble_uris(bucket=bucket, filenames=filenames)

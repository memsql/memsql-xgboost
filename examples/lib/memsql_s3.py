import io
import pickle
import tarfile
from xgboost import Booster
from typing import Optional, List, Dict
from boto3.session import Session
from memsql.common.database import Connection

from .memsql_udf import F, upload_xgb_to_memsql


def load_file_from_s3(s3_path: str, session: Session) -> bytes:
    PREFIX = 's3://'
    if not s3_path.startswith('s3://'):
        raise Exception(f's3_path should start with "{PREFIX}" (got s3_path="{s3_path}")')

    bucket, key = s3_path[len(PREFIX):].split('/', 1)
    s3 = session.resource('s3')
    obj = s3.Object(bucket, key)
    return obj.get()['Body'].read()


def load_xgboost_from_s3(s3_path: str, session: Session, model_file_name='xgboost-model') -> Optional[Booster]:
    model_tar_gz = load_file_from_s3(s3_path, session)
    tar = tarfile.open(fileobj=io.BytesIO(model_tar_gz), mode='r:gz')
    for member in tar.getmembers():
        if member.name == model_file_name:
            return pickle.loads(tar.extractfile(member).read())
    raise Exception(f'Could not find {model_file_name} in file "{s3_path}"')


def xgb_model_path_to_memsql(udf_name: str,
                             xgb_s3_path: str,
                             conn: Connection,
                             session: Session,
                             feature_names: List[str] = None,
                             func=F.SIGMOID,
                             allow_overwrite: bool = False) -> None:
    xgb = load_xgboost_from_s3(xgb_s3_path, session)
    upload_xgb_to_memsql(xgb, conn, udf_name,
                         func=func,
                         feature_names=feature_names,
                         allow_overwrite=allow_overwrite)

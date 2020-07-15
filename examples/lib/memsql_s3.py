import json
import tarfile
import boto3
import io
import pickle
from typing import List
from pandas import DataFrame
from xgboost.core import Booster


def export_as_csv(conn, dest, table, label, columns, config, creds, where='1=1'):
    columns = ','.join([label] + columns)
    config = json.dumps(config)
    creds = json.dumps(creds)
    query = f"SELECT {columns} FROM {table} WHERE {where} ORDER BY id INTO S3 '{dest}' CONFIG '{config}' CREDENTIALS '{creds}' FIELDS TERMINATED BY ','";
    conn.query(query)
    
    
def load_file_from_s3(s3_path):
    PREFIX = 's3://'
    if not s3_path.startswith('s3://'):
        raise Exception(f's3_path should start with "{PREFIX}" (got s3_path="{s3_path}")')
        
    bucket, key = s3_path[len(PREFIX):].split('/', 1)
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    return obj.get()['Body'].read()

    
def load_xgboost(s3_path):
    FILE_NAME = 'xgboost-model' 
    model_tar_gz = load_file_from_s3(s3_path)
    tar = tarfile.open(fileobj=io.BytesIO(model_tar_gz), mode='r:gz')
    for member in tar.getmembers():
        if member.name == FILE_NAME:
                return pickle.loads(tar.extractfile(member).read())
    raise Exception(f'Could not find {FILE_NAME} in file "{s3_path}"')

    
def xgb_to_memsql(xgb, features, conn):
    mm = {'f' + str(i): name for i, name in enumerate(features)}
    trees = split_trees(xgb.trees_to_dataframe())
    sqls = [tree_to_func_def(t, mm) for t in trees]
    sqls.append(tree_to_main_func(trees, features, mm))
    for s in sqls:
        assert 1 == conn.query(s)


def xgb_model_path_to_memsql(xgb_s3_path, features, conn):
    xgb = load_xgboost(xgb_s3_path)
    xgb_to_memsql(xgb, features, conn)

    
#####################################################

def split_trees(tdf: DataFrame) -> List[DataFrame]:
    trees_ids = set(tdf['Tree'])
    return [tdf[tdf['Tree'] == i] for i in trees_ids]


def tree_to_func_name(tree: DataFrame) -> str:
    tree_id = list(tree['Tree'])[0]
    return f"eval_tree_{tree_id}"


def format_sql_lines(lines: List[str], prefix: str) -> str:
    return f"\n{prefix}".join(lines)


def node_to_statement(node_id: str, tree: DataFrame, prefix: str, mm) -> str:
    prefix += '\t'
    node = tree[tree['ID'] == node_id].iloc[0]
    if node["Feature"] == "Leaf":
        return format_sql_lines([f"RETURN {node['Gain']};"], prefix)
    return format_sql_lines([
        f"IF {mm[node['Feature']]} < {node['Split']} THEN",
        node_to_statement(node["Yes"], tree, prefix, mm),
        "ELSE",
        node_to_statement(node["No"], tree, prefix, mm),
        "END IF;",
    ], prefix)


def tree_to_statements(tree: DataFrame, mm) -> str:
    used_nodes = set(tree[~tree["Yes"].isnull()]["Yes"]).union(set(tree[~tree["No"].isnull()]["No"]))
    roots = set(tree[~tree["ID"].isin(used_nodes)]["ID"])
    assert len(roots) == 1, f"expected to find one root, got {len(roots)}: {roots}"
    root = list(roots)[0]
    return node_to_statement(root, tree, "", mm)


def tree_to_func_def(tree: DataFrame, mm) -> str:
    features = sorted([mm[f] for f in set(tree['Feature']) if f != "Leaf"])
    args = ", ".join([f"{f} DOUBLE NOT NULL" for f in features])
    function_body = tree_to_statements(tree, mm)
    func_name = tree_to_func_name(tree)

    return format_sql_lines([
        f"CREATE OR REPLACE FUNCTION {func_name}({args}) RETURNS DOUBLE NOT NULL AS",
        "BEGIN",
        function_body,
        "END"
    ], "")

    
def tree_to_function_call(tree: DataFrame, mm) -> str:
    func_name = tree_to_func_name(tree)
    features = sorted([mm[f] for f in set(tree['Feature']) if f != "Leaf"])
    return f"{func_name}({', '.join(features)})"


def trees_to_functions_sum(trees: List[DataFrame], prefix, mm) -> str:
    return '+'.join([tree_to_function_call(t, mm) for t in trees])


def tree_to_main_func(trees: List[DataFrame], features, mm) -> str:
    #features = sorted([f for f in set(sum([list(tree['Feature']) for tree in trees], [])) if f != "Leaf"])
    args = ", ".join([f"{f} DOUBLE NOT NULL" for f in features])
    return format_sql_lines([
        f"CREATE OR REPLACE FUNCTION apply_trees({args}) RETURNS DOUBLE NOT NULL AS",
        "BEGIN",
        "\tRETURN SIGMOID(" + trees_to_functions_sum(trees, "\t", mm) + ");",
        "END"
    ], "")

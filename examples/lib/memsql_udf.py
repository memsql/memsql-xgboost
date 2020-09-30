import enum
from typing import List, Set, Iterable
from pandas import DataFrame
from xgboost import Booster
from memsql.common.database import Connection


class F(enum.Enum):
    SIGMOID = 1
    SUM = 2

    @staticmethod
    def apply(args: List[str], function):
        if function == F.SIGMOID:
            return f"SIGMOID({'+'.join(args)})"
        if function == F.SUM:
            return '+'.join(args)
        raise Exception(f"Unknown function type: {function}")


def backtick_escape(s: str) -> str:
    return '`' + s + '`'


def split_trees(tdf: DataFrame) -> List[DataFrame]:
    trees_ids = set(tdf['Tree'])
    return [tdf[tdf['Tree'] == i] for i in trees_ids]


def get_tree_features(tree: DataFrame) -> Set[str]:
    features = set(tree['Feature'])
    features.discard('Leaf')
    return features


def features_to_call(features: Iterable[str]) -> str:
    return ", ".join([backtick_escape(f) for f in sorted(features)])


def features_to_args(features: Iterable[str]) -> str:
    return ", ".join([backtick_escape(f) + " DOUBLE NOT NULL" for f in sorted(features)])


def tree_to_func_name(udf_name: str, tree: DataFrame) -> str:
    tree_id = list(tree['Tree'])[0]
    return f"{udf_name}_tree{tree_id}"


def node_to_statement(node_id: str, tree: DataFrame, prefix: str) -> str:
    prefix += '\t'
    node = tree[tree['ID'] == node_id].iloc[0]
    if node["Feature"] == "Leaf":
        return f"{prefix}RETURN {node['Gain']};"
    yes_branch = node_to_statement(node["Yes"], tree, prefix)
    no_branch = node_to_statement(node["No"], tree, prefix)
    return f"""{prefix}IF `{node['Feature']}` < {node['Split']} THEN
{prefix}{yes_branch}
{prefix}ELSE
{prefix}{no_branch}
{prefix}END IF;"""


def tree_to_statements(tree: DataFrame) -> str:
    used_nodes = set(tree[~tree["Yes"].isnull()]["Yes"]).union(set(tree[~tree["No"].isnull()]["No"]))
    roots = set(tree[~tree["ID"].isin(used_nodes)]["ID"])
    assert len(roots) == 1, f"expected to find one root, got {len(roots)}: {roots}"
    root = list(roots)[0]
    return node_to_statement(root, tree, "")


def tree_to_func_def(udf_name: str, allow_overwrite: bool, tree: DataFrame) -> str:
    features = get_tree_features(tree)
    args = features_to_args(features)
    function_body = tree_to_statements(tree)
    func_name = tree_to_func_name(udf_name, tree)
    or_replace = 'OR REPLACE' if allow_overwrite else ''
    return f"""CREATE {or_replace} FUNCTION {func_name}({args}) RETURNS DOUBLE NOT NULL AS
BEGIN
{function_body}
END"""


def tree_to_function_call(udf_name: str, tree: DataFrame) -> str:
    func_name = tree_to_func_name(udf_name, tree)
    features = get_tree_features(tree)
    return f"{func_name}({', '.join(sorted(features))})"


def trees_to_functions_sum(udf_name: str, trees: List[DataFrame]) -> str:
    return '+'.join([tree_to_function_call(udf_name, t) for t in trees])


def tree_to_main_func(udf_name: str, allow_overwrite: bool, trees: List[DataFrame], features: List[str], func: F) -> str:
    # Don't use features_to_args because we need arguments in the original order
    args = ", ".join([backtick_escape(f) + " DOUBLE NOT NULL" for f in features])
    or_replace = 'OR REPLACE' if allow_overwrite else ''
    return f"""CREATE {or_replace} FUNCTION {udf_name}({args}) RETURNS DOUBLE NOT NULL AS
BEGIN
    RETURN {F.apply([tree_to_function_call(udf_name, t) for t in trees], func)};
END"""


def upload_xgb_to_memsql(xgb: Booster,
                         conn: Connection,
                         udf_name: str,
                         func=F.SIGMOID,
                         feature_names: List[str] = None,
                         allow_overwrite: bool = False) -> None:
    if feature_names:
        xgb.feature_names = feature_names
    trees = split_trees(xgb.trees_to_dataframe())
    sqls = [tree_to_func_def(udf_name, allow_overwrite, t) for t in trees]
    sqls.append(tree_to_main_func(udf_name, allow_overwrite, trees, xgb.feature_names, func))
    for s in sqls:
        assert 1 == conn.query(s)

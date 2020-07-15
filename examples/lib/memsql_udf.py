import enum
from typing import List
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


def split_trees(tdf: DataFrame) -> List[DataFrame]:
    trees_ids = set(tdf['Tree'])
    return [tdf[tdf['Tree'] == i] for i in trees_ids]


def tree_to_func_name(tree: DataFrame) -> str:
    tree_id = list(tree['Tree'])[0]
    return f"eval_tree_{tree_id}"


def format_sql_lines(lines: List[str], prefix: str) -> str:
    return f"\n{prefix}".join(lines)


def node_to_statement(node_id: str, tree: DataFrame, prefix: str) -> str:
    prefix += '\t'
    node = tree[tree['ID'] == node_id].iloc[0]
    if node["Feature"] == "Leaf":
        return format_sql_lines([f"RETURN {node['Gain']};"], prefix)
    return format_sql_lines([
        f"IF `{node['Feature']}` < {node['Split']} THEN",
        node_to_statement(node["Yes"], tree, prefix),
        "ELSE",
        node_to_statement(node["No"], tree, prefix),
        "END IF;",
    ], prefix)


def tree_to_statements(tree: DataFrame) -> str:
    used_nodes = set(tree[~tree["Yes"].isnull()]["Yes"]).union(set(tree[~tree["No"].isnull()]["No"]))
    roots = set(tree[~tree["ID"].isin(used_nodes)]["ID"])
    assert len(roots) == 1, f"expected to find one root, got {len(roots)}: {roots}"
    root = list(roots)[0]
    return node_to_statement(root, tree, "")


def tree_to_func_def(tree: DataFrame) -> str:
    features = sorted([f for f in set(tree['Feature']) if f != "Leaf"])
    args = ", ".join([f"{f} DOUBLE NOT NULL" for f in features])
    function_body = tree_to_statements(tree)
    func_name = tree_to_func_name(tree)

    return format_sql_lines([
        f"CREATE OR REPLACE FUNCTION {func_name}({args}) RETURNS DOUBLE NOT NULL AS",
        "BEGIN",
        function_body,
        "END"
    ], "")


def tree_to_function_call(tree: DataFrame) -> str:
    func_name = tree_to_func_name(tree)
    features = sorted([f for f in set(tree['Feature']) if f != "Leaf"])
    return f"{func_name}({', '.join(features)})"


def trees_to_functions_sum(trees: List[DataFrame]) -> str:
    return '+'.join([tree_to_function_call(t) for t in trees])


def tree_to_main_func(trees: List[DataFrame], features: List[str], func: F) -> str:
    args = ", ".join([f"`{f}` DOUBLE NOT NULL" for f in features])
    return format_sql_lines([
        f"CREATE OR REPLACE FUNCTION apply_trees({args}) RETURNS DOUBLE NOT NULL AS",
        "BEGIN",
        f"\tRETURN {F.apply([tree_to_function_call(t) for t in trees], func)};",
        "END"
    ], "")


def upload_xgb_to_memsql(xgb: Booster, features: List[str], conn: Connection, func=F.SIGMOID) -> None:
    trees = split_trees(xgb.trees_to_dataframe())
    sqls = [tree_to_func_def(t) for t in trees]
    sqls.append(tree_to_main_func(trees, features, func))
    for s in sqls:
        assert 1 == conn.query(s)

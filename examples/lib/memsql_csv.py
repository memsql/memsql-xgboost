import json
from typing import List
from memsql.common.database import Connection


def load_csv_to_table(csv_path: str, table_name: str, features: List[str], conn: Connection):
    columns = ", ".join([f"`{f}` DOUBLE NOT NULL" for f in features])
    conn.query(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    load_csv_to_existing_table(csv_path, table_name, conn)


def load_csv_to_existing_table(csv_path: str, table_name: str, conn: Connection):
    if conn.query(f"SELECT COUNT(*) FROM `{table_name}`").rows[0][0] > 0:
        raise Exception(f"Table {table_name} is not empty")
    assert conn.query(f"LOAD DATA LOCAL INFILE '{csv_path}' INTO TABLE {table_name} "
                      f"FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES") > 0


def export_as_csv(conn, dest, table, label, columns, config, creds, where='1=1'):
    columns = ','.join(map(lambda s: f"`{s}`", [label] + columns))
    config = json.dumps(config)
    creds = json.dumps(creds)
    query = f"SELECT {columns} FROM {table} WHERE {where} ORDER BY id INTO S3 '{dest}' CONFIG '{config}' CREDENTIALS '{creds}' FIELDS TERMINATED BY ','"
    conn.query(query)

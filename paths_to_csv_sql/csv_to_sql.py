import sqlite3
import pandas as pd

def import_csv_to_sqlite(db_name, csv_file, table_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        df = pd.read_csv(csv_file)

        df.to_sql(table_name, conn, if_exists='replace', index=False)

        conn.commit()
        print(f"CSV file '{csv_file}' imported successfully into the table '{table_name}' in the database '{db_name}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()

import_csv_to_sqlite('train.db', 'C:\\Users\\zenklai\\train.csv', 'train')
import_csv_to_sqlite('test.db', 'C:\\Users\\zenklai\\test.csv', 'test')
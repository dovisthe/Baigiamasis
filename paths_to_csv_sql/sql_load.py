import sqlite3
import pandas as pd


def fetch_data_from_db(db_name, table_name):
    try:
        conn = sqlite3.connect(db_name)

        query = f"SELECT ClassId, Path FROM {table_name}" 
        data_df = pd.read_sql_query(query, conn)

        print(f"Data fetched successfully from the table '{table_name}' in the database '{db_name}'.")
        return data_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        conn.close()

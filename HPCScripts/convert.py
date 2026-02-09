#import abbu file DB and convert to csv
import pandas as pd
import os
import sqlite3
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

#first convert to xml?

def abbu_to_csv(abbu_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Connect to the ABBU SQLite database
    conn = sqlite3.connect(abbu_file)
    cursor = conn.cursor()

    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name in tqdm(tables, desc="Converting tables"):
        table_name = table_name[0]
        logging.info(f"Processing table: {table_name}")

        # Read the table into a DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        # Define output CSV file path
        csv_file_path = os.path.join(output_dir, f"{table_name}.csv")

        # Save DataFrame to CSV
        df.to_csv(csv_file_path, index=False)
        logging.info(f"Saved {table_name} to {csv_file_path}")

    # Close the database connection
    conn.close()
    logging.info("Conversion completed.")

abbu_file = r"C:\Users\mnand\Downloads\AddressBook-v22 (2).abcddb"

output_dir = r"C:\Users\mnand\Downloads\abbu_csv"

abbu_to_csv(abbu_file, output_dir)

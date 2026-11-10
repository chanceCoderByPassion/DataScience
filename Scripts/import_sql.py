import mysql.connector
from mysql.connector import Error
import os
from pathlib import Path


def import_sql_file(host, user, password, database, sql_file_path):
    connection = None
    try:
        # 1. Establish the connection to the MySQL server
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # 2. Read the entire content of the .sql file
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                sql_script = file.read()
            
            print(f"Executing {sql_file_path}...")
            
            # 3. Use multi=True to execute multiple statements sequentially
            # cursor.execute returns an iterator over the statement results
            results = cursor.execute(sql_script, multi=True)
            
            # 4. We must iterate through the results to ensure everything executes
            for result in results:
                if result.with_rows:
                    # If a query returns rows (like a SELECT statement), fetch them
                    result.fetchall()
            
            # 5. Commit the changes to the database
            connection.commit()
            print("Database imported successfully!")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        if connection:
            connection.rollback()
            
    finally:
        # 6. Clean up and close connections
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")


#!/usr/bin/env python3
import psutil
import time
import mysql.connector
import argparse

def main(param_names):
    cpu_usage = get_cpu_usage()
    disk_iops = get_disk_iops()
    print(f"cpu usage: {cpu_usage}, iops: {disk_iops}")
    
    cpu_threshold, iops_threshold = read_threshold_from_repo()
    iops_threshold = 30
    if cpu_usage > cpu_threshold and disk_iops > iops_threshold:
        new_param_values = read_parameters_from_repo(param_names)
        update_myqsql_parameters(param_names, new_param_values)


def get_cpu_usage(): return psutil.cpu_percent()

def get_disk_iops():
    disk_io_counters = psutil.disk_io_counters()
    read_ps = disk_io_counters.read_count / disk_io_counters.read_time
    write_ps = disk_io_counters.write_count / disk_io_counters.write_time

    starttime = time.time()
    io_counters_start = psutil.disk_io_counters()

    # Get the initial disk I/O counters for the entire system
    io_counters_start = psutil.disk_io_counters()


    # Wait for a short time
    time.sleep(1)
    endtime = time.time()
    # Get the updated disk I/O counters for the entire system
    io_counters_end = psutil.disk_io_counters()

    # Calculate the elapsed time between the two measurements
    elapsed_time = endtime - starttime

    # Calculate the input and output operation counts per second
    read_count_ps = (io_counters_end.read_count - io_counters_start.read_count) / elapsed_time
    write_count_ps = (io_counters_end.write_count - io_counters_start.write_count) / elapsed_time
    return read_count_ps + write_count_ps


def read_threshold_from_repo():
    cpu_threshold = fetch_column_from_config_repo("CPU_usage_percent")
    iops_threshold = fetch_column_from_config_repo("IOPS")
    return cpu_threshold, iops_threshold

def read_parameters_from_repo(columns):
    values = []
    for clm in columns:
        values.append(fetch_column_from_config_repo(clm))
    return values


def update_myqsql_parameters(param_names, param_values):
    for name, value in param_names, param_values:
        print(f"{name}: {value}")
    return
    db=mysql.connector.connect(host="localhost", user=args.user, password=args.password)
    cursor=db.cursor()

    for name, value in param_names, param_values:
        set_param_query = f"SET GLOBAL {name} = {value}"
        cursor.execute(set_param_query)

    db.commit()

    cursor.close()
    db.close
    return


def fetch_column_from_config_repo(column, host="20.222.144.239", user="test", password="test", database="config_repo"):
    db=mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor=db.cursor()

    query = f"SELECT {column} from param_set"
    cursor.execute(query)
    result = cursor.fetchone()

    cursor.close()
    db.close()
    
    return result[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connect to MySQL')
    parser.add_argument('--user', metavar='user', type=str,
                        help='name of user',
                        default="root")
    parser.add_argument('--password', metavar='pw', type=str,
                        help='password for the user',
                        default="")
    args = parser.parse_args()
    parameters = ["innodb_buffer_pool_size", "innodb_io_capacity"]
    while True:
        main(parameters)
        time.sleep(1)
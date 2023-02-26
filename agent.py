#!/usr/bin/env python3
import psutil
import time
import mysql.connector
import os

def main():
    print("Agent Started")
    login_info = {"user": "root", "password": "root_passwd"}
    parameters = ["innodb_buffer_pool_size", "innodb_io_capacity"]
    sleeptime = 1

    start = time.time()
    while True:
        done_tuned = autotune(login_info, parameters)
        if done_tuned:
            end = time.time()
            tune_time = end - start
            output_result(tune_time)
            exit()
        time.sleep(sleeptime)

def output_result(tune_time):
    print(f"agent tuned mysql in {tune_time} seconds")
    os.system("mkdir -p /home/azureuser/AutoD/result")
    os.system(f"echo agent tuned mysql in {tune_time} seconds > /home/azureuser/AutoD/result/agent.log")

def autotune(login_info, param_names):
    cpu_usage = get_cpu_usage()
    disk_iops = get_disk_iops()
    print(f"cpu usage: {cpu_usage}, iops: {disk_iops}")
    
    cpu_threshold, iops_threshold = read_threshold_from_repo()
    if cpu_usage > cpu_threshold and disk_iops > iops_threshold:
        new_param_values = read_parameters_from_repo(param_names)
        update_myqsql_parameters(login_info, param_names, new_param_values)
        return True
    return False


def get_cpu_usage(): return psutil.cpu_percent()

def get_disk_iops():
    starttime = time.time()
    io_counters_start = psutil.disk_io_counters()

    time.sleep(1)

    endtime = time.time()
    io_counters_end = psutil.disk_io_counters()

    elapsed_time = endtime - starttime

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


def update_myqsql_parameters(login_info, param_names, param_values):
    # for name, value in zip(param_names, param_values):
    #     print(f"{name}: {value}")

    db=mysql.connector.connect(host="localhost", user=login_info["user"], password=login_info["password"])
    cursor=db.cursor()

    for name, value in zip(param_names, param_values):
        set_param_query = f"SET GLOBAL {name}={value}"
        show_param_query = f"SHOW GLOBAL VARIABLES LIKE \"{name}\""
        print(set_param_query)
        cursor.execute(set_param_query)
        cursor.execute(show_param_query)
        print(cursor.fetchone())
        

    db.commit()

    cursor.close()
    db.close


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
    main()
import argparse
import collections
import csv
import datetime
import logging
from IPython import embed
import time

import requests
import urllib3

# create a keyvalue class
class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
          
        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value

def set_query_in_params(params, query):
    params['query'] = query
    return params

def prepare_params_for_GET(args):
    step = args.step
    # 引数の開始時間と終了時間をUNIX時刻に変換
    start_str = args.start
    end_str = args.end
    minute = 60
    if start_str == None:
        end_unix = time.time()
        start_unix = end_unix - 5*minute
    else:
        start_dt = datetime.datetime.strptime(start_str, '%Y%m%d-%H%M')
        end_dt = datetime.datetime.strptime(end_str, '%Y%m%d-%H%M')
        start_unix = start_dt.timestamp()
        end_unix = end_dt.timestamp()
    params = {'query': '',
              'start': start_unix,
              'end': end_unix,
              'step': step}
    return params

def send_requests(args, queries, url):
    results = []
    params = prepare_params_for_GET(args)
    for query in queries:
        request = set_query_in_params(params, query)
        response = requests.get(url, verify=False, params=request)
        response.raise_for_status()
        result = response.json()['data']['result']
        results.append(result)
    return results

def output_csv(filepath, results, queryNames):
    # 時刻毎のデータの辞書を用意する
    time_series = collections.defaultdict(dict)
    for (result, queryName) in zip(results, queryNames):
        try:
            assert len(result) == 1
        except AssertionError:
            print(queryName)
            continue
        json = result[0]
        for value in json['values']:
            # timestampを辞書のキーにすることで同じtimestampのデータをまとめる
            # defaultdictを使うことでキーがなくてもKeyErrorにならない
            time_series[value[0]][queryName] = value[1]

    fieldnames = ['timestamp']
    fieldnames.extend(queryNames)

    # csvファイルに保存する
    with open(filepath, 'w') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # 辞書から時間毎のデータを取り出してループする
        for timestamp, values in time_series.items():
            # 行に時間の列を追加
            row = {'timestamp': datetime.datetime.fromtimestamp(timestamp)}
            # valuesは以下のような辞書
            # {'query1': '2.467225521553685',
            #  'query2': '1.5932590068361583'},
            for queryName in queryNames:
                try:
                    row[queryName] = values[queryName]
                except KeyError:
                    row[queryName] = ''
            writer.writerow(row)

def add_column(filepath, label, value):
    # input file name
    file_name = filepath

    # label name and value
    label_name = label
    label_value = value

    # read input file
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # add label column
    data[0].append(label_name)
    for row in data[1:]:
        row.append(label_value)

    # write output file
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main(args):
    osMetrics = ['CPU Usage %', 
                 'memory usage %', 
                 'disk IOPS', 
                 'disk busy %', 
                 'disk usage gb', 
                 'network IOPS', 
                 'swap/s']
    osQueries = ['sum(avg without(cpu) (rate(node_cpu_seconds_total{mode!="idle"}[1m]))) * 100', #bottom
                 '(node_memory_MemTotal_bytes - node_memory_MemFree_bytes - node_memory_Buffers_bytes - node_memory_Cached_bytes - node_memory_SReclaimable_bytes) / node_memory_MemTotal_bytes * 100', #free -b
                 'sum(rate(node_disk_reads_completed_total[1m]) + rate(node_disk_writes_completed_total[1m]))', #iostat -x
                 'sum(rate(node_disk_io_time_seconds_total[1m])) * 100', #iostat -x
                 '(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}) / 1024^3',
                 'sum(rate(node_network_receive_packets_total[1m]) + rate(node_network_transmit_packets_total[1m]))', #sar -n DEV
                 'rate(node_vmstat_pswpin[1m]) + rate(node_vmstat_pswpout[1m])'
                 ]

    innodbMetrics = ['innodb reads/s',
                     'innodb writes/s', 
                     'innodb buffer pool hit rate',
                     'innodb I/O (MB/s)',
                     'innodb_buffer_pool',
                     'innodb_io_capacity']
    innodbQueries = ['rate(mysql_global_status_innodb_data_reads[1m])',
                     'rate(mysql_global_status_innodb_data_writes[1m])',
                     '(1 - rate(mysql_global_status_innodb_buffer_pool_reads[1m]) / rate(mysql_global_status_innodb_buffer_pool_read_requests[1m])) * 100',
                     '(rate(mysql_global_status_innodb_data_read[1m]) + rate(mysql_global_status_innodb_data_written[1m])) / 1024^2',
                     'mysql_global_variables_innodb_buffer_pool_size',
                     'mysql_global_variables_innodb_io_capacity']

    mysqlMetrics = ['threads running/s', 
                    'handler call/s', 
                    'num slow queries', 
                    'created tmp tables/s', 
                    'sort merge/s', 'scan/s', 
                    'avg lock time/s', 'query rate',
                    'query response time', 'tps']
    mysqlQueries = ['rate(mysql_global_status_threads_running[1m])',
                    'sum(rate(mysql_global_status_handlers_total[1m]))',
                    'mysql_global_status_slow_queries',
                    'rate(mysql_global_status_created_tmp_tables[1m])',
                    'rate(mysql_global_status_sort_merge_passes[1m])',
                    'rate(mysql_global_status_select_scan[1m])',
                    'rate(mysql_global_status_innodb_row_lock_time_avg[1m])',
                    'sum(rate(mysql_global_status_commands_total[1m]))',                
                    'avg(rate(mysql_perf_schema_events_statements_seconds_total[1m]))',
                    'sum(rate(mysql_global_status_commands_total{command=~"(commit|rollback)"}[1m])) without (command)']

    queryName = osMetrics + innodbMetrics + mysqlMetrics
    queries = osQueries + innodbQueries + mysqlQueries

    assert len(queryName) == len(queries)
    url = 'http://20.222.144.239:9090/api/v1/query_range'
    results = send_requests(args, queries, url)
    output_csv(args.filename, results, queryName)
    if args.configs is not None:
        for key, value in args.configs.items():
            add_column(args.filename, key, value)


if __name__ == "__main__":
    formatter = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.INFO, format=formatter)
    logger = logging.getLogger(__name__)

    # 警告を非表示にする
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description='csvに出力します。')
    parser.add_argument('-f', '--filename',
                        action='store',
                        type=str,
                        default="metrics.csv",
                        help='出力先のファイル名を指定します')
    parser.add_argument('--start',
                        action='store',
                        type=str,
                        help='データの開始時間を指定します（例）20190101-1000')
    parser.add_argument('--end',
                        action='store',
                        type=str,
                        help='データの終了時間を指定します（例）20190102-1000')
    parser.add_argument('--step',
                        action='store',
                        type=str,
                        default='5s',
                        help='データポイントの間隔（秒）を指定します')
    parser.add_argument('--configs', 
                        nargs='*', 
                        action=keyvalue)
    args = parser.parse_args()
    main(args)
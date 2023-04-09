import os
import argparse
from time import sleep

def main():
  workloads = ["oltp_read_write", "oltp_read_only", "oltp_write_only"]
  threads = [1, 10, 60]
  buf_sizes = [134217728, 1073741824, 16777216]
  io_caps = [100, 200, 400]
  count = 0
  path = args.path + "/train"
  for workload in workloads:
    for thread in threads:
      for size in buf_sizes:
        for cap in io_caps:
          os.system(f"mysql -h {args.host} -u {args.user} -p{args.password} -e \"set global innodb_buffer_pool_size={size}; set global innodb_io_capacity={cap}\"")
          os.system(f"sysbench --db-driver=mysql --mysql-host={args.host} --mysql-user={args.user} --mysql-password={args.password} --mysql-db={args.db} --tables={args.tablenum} --table_size={args.tablesize} --threads={thread} --time={args.time} {workload} run")
          os.system(f"python3 exporter.py --configs label={workload} tablesize={args.tablesize} -f {path}/{count}.csv")
          os.system(f"mysql -h {args.host} -u {args.user} -p{args.password} -e \"PURGE BINARY LOGS BEFORE NOW();\"")
          # count += 1
          # if count > (len(workloads) * len(threads) * len(buf_sizes) * len(io_caps) * 0.7): path = args.path + "/test"
          sleep(args.sleep)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='make training data csv')
  parser.add_argument('--host',
                      action='store',
                      type=str,
                      default="20.222.144.239",
                      help='mysql host')
  parser.add_argument('--user',
                      action='store',
                      type=str,
                      default="test",
                      help='mysql user')
  parser.add_argument('--password',
                      action='store',
                      type=str,
                      default="test",
                      help='mysql password')
  parser.add_argument('--db',
                      action='store',
                      type=str,
                      default="test",
                      help='mysql db')
  parser.add_argument('--tablesize',
                      action='store',
                      type=str,
                      default="1000000",
                      help='table size')
  parser.add_argument('--tablenum',
                      action='store',
                      type=str,
                      default="4",
                      help='table num')
  parser.add_argument('-t', '--time',
                      action='store',
                      type=str,
                      default="300",
                      help='test duration')
  parser.add_argument('-p', '--path',
                      action='store',
                      type=str,
                      required=True,
                      help='test duration')
  parser.add_argument('-s', '--sleep',
                      action='store',
                      type=int,
                      default=80,
                      help='sleep duration')
  args = parser.parse_args()
  main()
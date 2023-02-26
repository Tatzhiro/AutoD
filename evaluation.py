import os
import argparse
from time import sleep
import mysql.connector
import agent

def init_db(db_name):
  db=mysql.connector.connect(host="localhost", user=args.user, password=args.password)
  cursor=db.cursor()
  drop_query = f"DROP DATABASE IF EXISTS {db_name};"
  create_query = f"CREATE DATABASE {db_name};"
  cursor.execute(drop_query)
  # cursor.execute("PURGE BINARY LOGS BEFORE NOW();")
  cursor.execute(create_query)
  cursor.execute("SET GLOBAL innodb_buffer_pool_size=134217728;")
  cursor.execute("SET GLOBAL innodb_io_capacity=200;")

  db.commit()

  cursor.close()
  db.close

def prepare_bench():
  os.system(f"sysbench --db-driver=mysql --mysql-host={args.host} --create_secondary=on \
            --mysql-user={args.user} --mysql-password={args.password} --mysql-db={args.db} --tables={args.tablenum} \
            --table_size={args.tablesize} --threads=10 oltp_read_write prepare")

def benchmark():
  os.system("mkdir -p /home/azureuser/AutoD/result")
  os.system(f"sysbench --db-driver=mysql --mysql-host={args.host} --mysql-user={args.user} \
             --mysql-password={args.password} --mysql-db={args.db} --tables={args.tablenum} \
            --table_size={args.tablesize} --threads={args.threadnum} --time={args.time} \
            oltp_read_write run 2>&1 | tee /home/azureuser/AutoD/result/{args.filename}.log")

def main():
  init_db(args.db)
  prepare_bench()
  pid = os.fork()
  if pid > 0 :
    benchmark()
  else : 
    if args.filename == "autotune":
      agent.main()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='make training data csv')
  parser.add_argument('--host',
                      action='store',
                      type=str,
                      default="localhost",
                      help='mysql host')
  parser.add_argument('--user',
                      action='store',
                      type=str,
                      default="root",
                      help='mysql user')
  parser.add_argument('--password',
                      action='store',
                      type=str,
                      default="root_passwd",
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
                      default="10",
                      help='table num')
  parser.add_argument('--time',
                      action='store',
                      type=str,
                      default="300",
                      help='test duration')
  parser.add_argument('-t', '--threadnum',
                      action='store',
                      type=str,
                      default="60",
                      help='thread num')
  parser.add_argument('-s', '--sleep',
                      action='store',
                      type=int,
                      default=5,
                      help='sleep duration')
  parser.add_argument('-f', '--filename',
                      action='store',
                      type=str,
                      default="autotune",
                      help='output filename')
  args = parser.parse_args()
  main()
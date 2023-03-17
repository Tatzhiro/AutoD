import pandas as pd
import argparse

def calculate_num_rows():
  minutes = args.time * 60
  step_second = args.steps
  num_rows = minutes / step_second
  return num_rows

def average_column(filename, column_name):
  df = pd.read_csv(filename)
  column = df[column_name]
  rows = calculate_num_rows()
  average = column.loc[0:rows-1].mean()
  return average


def main():
  average = average_column(args.filename, args.clmname)
  print(average)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='make training data csv')
  parser.add_argument('-f', '--filename',
                      action='store',
                      type=str,
                      default="metrics.csv",
                      help='読み込むファイル名を指定します')
  parser.add_argument('-c', '--clmname',
                      action='store',
                      type=str,
                      default="disk IOPS",
                      help='column name')
  parser.add_argument('-s', '--steps',
                      action='store',
                      type=int,
                      default=5,
                      help='interval of datapoints in seconds')
  parser.add_argument('-t', '--time',
                      action='store',
                      type=int,
                      default=1,
                      help='average over how many minutes')
  args = parser.parse_args()
  main()
import pandas as pd
import argparse

def average_column(filename, column_name):
  df = pd.read_csv(filename)
  column = df[column_name]
  average = column.mean()
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
                      help='出力先のファイル名を指定します')
  parser.add_argument('-c', '--clmname',
                      action='store',
                      type=str,
                      default="disk IOPS",
                      help='column name')
  args = parser.parse_args()
  main()
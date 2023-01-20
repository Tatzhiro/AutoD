import pandas as pd
import glob
import argparse

def main():
  # Find all CSV files in the current directory
  csv_files = glob.glob(args.path + "/*.csv")

  # Create an empty list to store the DataFrames
  dataframes = []

  # Read in the contents of each CSV file and store it in a DataFrame
  for file in csv_files:
      df = pd.read_csv(file)
      dataframes.append(df)

  # Concatenate the DataFrames along the rows
  result = pd.concat(dataframes, ignore_index=True)

  # Write the result to a new CSV file
  result.to_csv(args.path + "/" + args.filename, index=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='csvに出力します。')
  parser.add_argument('-f', '--filename',
                      required=True,
                      action='store',
                      type=str,
                      help='出力先のファイル名を指定します')
  parser.add_argument('-p', '--path',
                      required=True,
                      action='store',
                      type=str,
                      default=".",
                      help='path to csv')
  args = parser.parse_args()
  main()
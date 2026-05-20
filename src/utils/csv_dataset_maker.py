import csv
import pandas as pd


def make_dataset(source, dest, label, n_train, n_test, n_val):
        df = pd.read_csv(source + label + ".csv")

        df_train = df.iloc[:n_train-1]
        df_test = df.iloc[n_train-1:n_train+n_test-2]
        df_val = df.iloc[n_train+n_test-2:n_train+n_test+n_val-3]

        df_train.to_csv(dest+ "/Train/" + label + "_train.csv", index=False)
        df_test.to_csv(dest + "/Test/" + label + "_test.csv", index=False)
        df_val.to_csv(dest + "/Validation/" + label + "_val.csv", index=False)

for d in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] :
    make_dataset("_csv/Train/", "Dataset_csv/", d, 1800, 600, 600)
    print(d, " ok")



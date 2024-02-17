import os
import pandas as pd


# Add as a class so we can initiate any needed connections in the future to load data
# otherwise can leave it to be a function.
class CSVDataLoader:
    def __init__(self):
        self.df = None

    def load_data(self, filepath, header="infer", names=None):
        """
        Loads csv/tsv format files from specified path.

        Parameters
        ----------
        filepath : str
            path to the file to be loaded
        header : str
            row (0-indexed) to use as header or None if there is no header
        names : list
            list of column names to use
        """
        _, file_extension = os.path.splitext(filepath)
        file_extension = file_extension.lower()

        if file_extension in [".csv"]:
            sep = ","
        elif file_extension in [".tsv"]:
            sep = '\t'
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Use 'csv' or 'tsv'."
                )
        self.df = pd.read_csv(filepath, header=header, sep=sep, names=names)
        print(f"{filepath} file loaded successfully.")

        return self.df

import pytest
from recommender.dataloader import CSVDataLoader


@pytest.fixture
def csv_loader():
    return CSVDataLoader()


def test_load_csv_file(csv_loader):
    filepath = "tests/test_csv.csv"
    header = None
    names = ["tags"]

    df = csv_loader.load_data(filepath, header=header, names=names)

    assert df is not None

    assert "tags" in df.columns


def test_load_tsv_file(csv_loader):
    filepath = "tests/test_tsv.tsv"
    header = "infer"

    df = csv_loader.load_data(filepath, header=header)

    assert df is not None

    assert "keywords" in df.columns


def test_invalid_file_extension(csv_loader):
    filepath = "test_pq.pq"
    header = "infer"
    names = None

    with pytest.raises(ValueError):
        csv_loader.load_data(filepath, header=header, names=names)

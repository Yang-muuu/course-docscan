import pytest

from docscan.samples import make_samples


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic")
    make_samples(folder, 42)
    return folder

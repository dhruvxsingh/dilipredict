import pytest
from dilipredict.image_loader import ImageLoader

@pytest.fixture
def image_loader():
    return ImageLoader()

from src.main import ImageProcessor, Resnet18Predictor, ImageLoader
import pytest
from PIL import Image


@pytest.fixture
def example_image_filename():
    return "test.jpg"


@pytest.fixture
def example_image():
    return Image.new("RGB", (100, 100))


@pytest.fixture
def example_image_path(tmpdir, example_image_filename, example_image):
    tmpdir.mkdir("images")
    img_path = tmpdir.join(f"images/{example_image_filename}")
    example_image.save(img_path)
    return str(img_path)


# Fixture for ImageLoader
@pytest.fixture
def image_loader(tmpdir):
    return ImageLoader(image_dir=str(tmpdir.join("images")))


# Fixture for ImageProcessor
@pytest.fixture
def image_processor():
    return ImageProcessor(output_image_size=256)


# Fixture for Resnet18Predictor
@pytest.fixture
def resnet18_predictor():
    return Resnet18Predictor()

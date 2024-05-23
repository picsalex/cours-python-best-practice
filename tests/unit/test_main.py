from unittest.mock import patch
from PIL import Image
import torch


class TestMain:
    def test_load_images(
        self, image_loader, example_image_path, example_image_filename, example_image
    ):
        # Mock os.listdir to return the example image path
        with patch("os.listdir", return_value=[example_image_filename]):
            images = image_loader.load_images()
            assert len(images) == 1
            assert images[0].size == example_image.size

    def test_apply_preprocessing(self, image_processor, example_image_path):
        img = Image.open(example_image_path)
        processed = image_processor.apply_preprocessing([img])
        assert len(processed) == 1
        assert processed[0].size() == (3, 256, 256)

    def test_preprocess_image(self, image_processor, example_image_path):
        img = Image.open(example_image_path)
        tensor = image_processor._preprocess_image(img)
        assert tensor.size() == (3, 256, 256)

    def test_predict(self, resnet18_predictor):
        mock_tensor = torch.zeros((3, 256, 256), dtype=torch.float32)

        with patch.object(
            resnet18_predictor.model, "forward", return_value=torch.tensor([[0.1, 0.9]])
        ):
            prediction = resnet18_predictor.predict([mock_tensor])
            assert prediction == [1]

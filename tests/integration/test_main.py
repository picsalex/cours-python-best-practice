from unittest.mock import patch

import torch
from src.main import ImageLoader


class TestMain:
    def test_full_flow(
        self, tmpdir, example_image, image_processor, resnet18_predictor
    ):
        image_path = tmpdir.join("test.jpg")
        example_image.save(image_path)

        loader = ImageLoader(image_dir=str(tmpdir))
        images = loader.load_images()

        preprocessed_tensor = image_processor.apply_preprocessing(images)

        with patch.object(
            resnet18_predictor.model, "forward", return_value=torch.tensor([[0.1, 0.9]])
        ):
            results = resnet18_predictor.predict(preprocessed_tensor)
            assert results == [1]

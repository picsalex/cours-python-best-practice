import os
from typing import List

from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter


class ImageLoader:
    def __init__(self, image_dir: str):
        self.image_dir = image_dir

    def load_images(self) -> List[Image.Image]:
        """
        Load images from the image directory

        Returns:
            List of loaded images
        """
        result = []

        for file_path in os.listdir(self.image_dir):
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                result.append(Image.open(os.path.join(self.image_dir, file_path)))

        return result


class ImageProcessor:

    def __init__(self, output_image_size: int):
        self.output_image_size = output_image_size

    def apply_preprocessing(self, images_list: List[Image.Image]) -> List[torch.Tensor]:
        """
        Apply preprocessing to the list of images.

        Args:
            images_list: List of images to be preprocessed

        Returns:
            List of preprocessed images
        """
        result = []
        for image in images_list:
            result.append(self._preprocess_image(image=image))

        return result

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the image

        Args:
            image: Image to be preprocessed

        Returns:
            Preprocessed image
        """
        t = transforms.Compose(
            [
                transforms.Resize((self.output_image_size, self.output_image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        return t(image)


class Resnet18Predictor:
    def __init__(self):
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.eval()

    def predict(self, tensor_list: List[torch.Tensor]) -> List[int]:
        """
        Predict the class of the image

        Args:
            tensor_list: List of preprocessed images

        Returns:
            List of predicted classes
        """
        result = []

        for tensor in tensor_list:
            result.append(self._make_prediction(tensor=tensor))

        return result

    def _make_prediction(self, tensor: torch.Tensor) -> int:
        """
        Make prediction on a single image

        Args:
            tensor: Preprocessed image

        Returns:
            Predicted class
        """
        prediction = self.model(tensor.unsqueeze(0))
        return torch.argmax(prediction, dim=1).item()


if __name__ == "__main__":
    writer = SummaryWriter("tensorboard/runs/image_classification")

    loader = ImageLoader(image_dir="data/")
    images = loader.load_images()

    processor = ImageProcessor(256)
    preprocessed_tensor = processor.apply_preprocessing(images)

    predictor = Resnet18Predictor()
    results = predictor.predict(tensor_list=preprocessed_tensor)

    for i, tensor in enumerate(preprocessed_tensor):
        writer.add_image(f"{results[i]}", tensor, 0)

    writer.close()

    print(f"Predicted {len(results)} images: {results}")

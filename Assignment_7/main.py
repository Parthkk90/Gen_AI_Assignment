"""
Basic Multimodal Image Captioning System using CLIP
=====================================================
This module implements a multimodal image captioning system using OpenAI's CLIP
(Contrastive Language-Image Pre-Training) model from Hugging Face Transformers.

CLIP learns visual concepts from natural language supervision, enabling zero-shot
image classification and captioning by ranking candidate text descriptions against
images using cosine similarity.

Key Features:
    - Zero-shot image captioning via candidate caption ranking
    - Image-to-text and text-to-image similarity scoring
    - Image categorization into predefined categories
    - Batch processing of multiple images
"""

import torch
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Tuple, Union
from transformers import CLIPProcessor, CLIPModel


class CLIPImageCaptioner:
    """
    A multimodal image captioning system using CLIP.

    CLIP uses contrastive learning to align image and text representations
    in a shared embedding space. This allows zero-shot caption selection
    by computing similarity between image embeddings and candidate text
    embeddings, then selecting the most similar caption.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP image captioning system.

        Args:
            model_name: The Hugging Face CLIP model to use.
                        Default is 'openai/clip-vit-base-patch32'.
        """
        print(f"Loading CLIP model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP model loaded successfully!\n")

    def load_image(self, image_source: str) -> Image.Image:
        """
        Load an image from a URL or local file path.

        Args:
            image_source: A URL string or local file path to the image.

        Returns:
            A PIL Image object.

        Raises:
            ValueError: If the image cannot be loaded from the given source.
        """
        try:
            if image_source.startswith(("http://", "https://")):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from '{image_source}': {e}")

    def generate_caption(
        self, image: Image.Image, candidate_captions: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a caption for an image by ranking candidate captions.

        Uses CLIP to compute similarity scores between the image and each
        candidate caption, returning the best match along with all scores.

        Args:
            image: A PIL Image object to caption.
            candidate_captions: A list of candidate text descriptions.

        Returns:
            A dictionary containing:
                - best_caption: The highest-scoring candidate caption.
                - confidence: The similarity score of the best caption (0-1).
                - all_scores: List of (caption, score) tuples sorted by score.
        """
        inputs = self.processor(
            text=candidate_captions, images=image, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probabilities = logits_per_image.softmax(dim=1).squeeze()

        scores = probabilities.tolist()
        if isinstance(scores, float):
            scores = [scores]

        caption_scores = list(zip(candidate_captions, scores))
        caption_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "best_caption": caption_scores[0][0],
            "confidence": caption_scores[0][1],
            "all_scores": caption_scores,
        }

    def compare_images(
        self, images: List[Image.Image], caption: str
    ) -> List[Tuple[int, float]]:
        """
        Rank multiple images by their similarity to a given text caption.

        Args:
            images: A list of PIL Image objects.
            caption: The text description to compare against.

        Returns:
            A list of (image_index, similarity_score) tuples sorted by score
            in descending order.
        """
        inputs = self.processor(
            text=[caption], images=images, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_text = outputs.logits_per_text
            probabilities = logits_per_text.softmax(dim=1).squeeze()

        scores = probabilities.tolist()
        if isinstance(scores, float):
            scores = [scores]

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores

    def analyze_image(
        self, image: Image.Image, categories: List[str]
    ) -> Dict[str, float]:
        """
        Categorize an image into predefined categories using CLIP.

        Uses zero-shot classification to determine which category best
        describes the image content.

        Args:
            image: A PIL Image object to categorize.
            categories: A list of category labels (e.g., ["cat", "dog", "car"]).

        Returns:
            A dictionary mapping each category to its similarity score (0-1),
            sorted by score in descending order.
        """
        prefixed_categories = [f"a photo of {cat}" for cat in categories]

        inputs = self.processor(
            text=prefixed_categories, images=image, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probabilities = logits_per_image.softmax(dim=1).squeeze()

        scores = probabilities.tolist()
        if isinstance(scores, float):
            scores = [scores]

        category_scores = dict(zip(categories, scores))
        category_scores = dict(
            sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return category_scores

    def batch_caption(
        self, image_sources: List[str], candidate_captions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images and generate captions for each.

        Args:
            image_sources: A list of image URLs or file paths.
            candidate_captions: A list of candidate text descriptions
                                to rank for each image.

        Returns:
            A list of result dictionaries, one per image, each containing
            the best caption, confidence score, and all scores.
        """
        results = []
        for idx, source in enumerate(image_sources):
            print(f"  Processing image {idx + 1}/{len(image_sources)}...")
            try:
                image = self.load_image(source)
                result = self.generate_caption(image, candidate_captions)
                result["source"] = source
                result["status"] = "success"
            except Exception as e:
                result = {
                    "source": source,
                    "status": "error",
                    "error": str(e),
                }
            results.append(result)

        return results

    def get_image_text_similarity(
        self, image: Image.Image, text: str
    ) -> float:
        """
        Compute the raw cosine similarity between an image and a text.

        Args:
            image: A PIL Image object.
            text: A text string.

        Returns:
            The cosine similarity score (higher means more similar).
        """
        inputs = self.processor(
            text=[text], images=image, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(input_ids=inputs["input_ids"],
                                                          attention_mask=inputs["attention_mask"])

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        return similarity


# ==================== HELPER FUNCTIONS ====================


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_caption_results(result: Dict[str, Any]):
    """Pretty-print caption generation results."""
    print(f"\n  Best Caption : {result['best_caption']}")
    print(f"  Confidence   : {result['confidence']:.4f}")
    print("\n  All Candidate Scores:")
    for caption, score in result["all_scores"]:
        bar = "█" * int(score * 40)
        print(f"    {score:.4f} | {bar} | {caption}")


def print_category_results(category_scores: Dict[str, float]):
    """Pretty-print image categorization results."""
    print("\n  Category Scores:")
    for category, score in category_scores.items():
        bar = "█" * int(score * 40)
        print(f"    {score:.4f} | {bar} | {category}")


# ==================== MAIN DEMONSTRATION ====================


def main():
    """
    Main function to demonstrate the CLIP image captioning system.

    Showcases:
        1. Single image captioning with candidate ranking
        2. Image categorization into predefined categories
        3. Image-to-text similarity comparison
        4. Multi-image comparison against a caption
        5. Batch captioning of multiple images
    """

    # Initialize the CLIP captioner
    captioner = CLIPImageCaptioner()

    # ----- Sample images (publicly available) -----
    sample_images = {
        "dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "city": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1200px-New_york_times_square-terabass.jpg",
    }

    # ===== DEMO 1: Single Image Captioning =====
    print_section_header("DEMO 1: Single Image Captioning")
    print("\n  Loading a dog image and ranking candidate captions...\n")

    try:
        dog_image = captioner.load_image(sample_images["dog"])
        print("  Image loaded successfully!")

        candidate_captions = [
            "a photo of a dog playing in the park",
            "a photo of a cat sitting on a couch",
            "a photo of a yellow labrador retriever",
            "a photo of a city skyline at night",
            "a photo of a bird flying in the sky",
            "a photo of a person riding a bicycle",
        ]

        result = captioner.generate_caption(dog_image, candidate_captions)
        print_caption_results(result)
    except Exception as e:
        print(f"  Error in Demo 1: {e}")

    # ===== DEMO 2: Image Categorization =====
    print_section_header("DEMO 2: Image Categorization (Zero-Shot)")
    print("\n  Categorizing the dog image into predefined categories...\n")

    try:
        categories = [
            "dog", "cat", "bird", "car", "building",
            "food", "landscape", "person", "flower", "airplane"
        ]

        category_scores = captioner.analyze_image(dog_image, categories)
        print_category_results(category_scores)
    except Exception as e:
        print(f"  Error in Demo 2: {e}")

    # ===== DEMO 3: Image-Text Similarity =====
    print_section_header("DEMO 3: Image-Text Similarity Scores")
    print("\n  Computing raw cosine similarity between image and text...\n")

    try:
        texts_to_compare = [
            "a golden retriever dog",
            "a domestic cat",
            "a red sports car",
            "a beautiful sunset over the ocean",
        ]

        print("  Similarity Scores (Dog Image vs Text):")
        for text in texts_to_compare:
            similarity = captioner.get_image_text_similarity(dog_image, text)
            bar = "█" * int(max(0, similarity) * 30)
            print(f"    {similarity:.4f} | {bar} | {text}")
    except Exception as e:
        print(f"  Error in Demo 3: {e}")

    # ===== DEMO 4: Multi-Image Comparison =====
    print_section_header("DEMO 4: Multi-Image Comparison")
    print("\n  Loading multiple images and ranking them against a caption...\n")

    try:
        images = []
        image_labels = []
        for label, url in sample_images.items():
            try:
                img = captioner.load_image(url)
                images.append(img)
                image_labels.append(label)
                print(f"  Loaded: {label} image")
            except Exception as e:
                print(f"  Failed to load {label} image: {e}")

        if len(images) >= 2:
            test_caption = "a photo of a cute animal"
            print(f"\n  Ranking images for caption: \"{test_caption}\"\n")

            rankings = captioner.compare_images(images, test_caption)
            print("  Image Rankings:")
            for rank, (idx, score) in enumerate(rankings, 1):
                bar = "█" * int(score * 40)
                print(f"    #{rank} | {score:.4f} | {bar} | {image_labels[idx]}")
    except Exception as e:
        print(f"  Error in Demo 4: {e}")

    # ===== DEMO 5: Batch Captioning =====
    print_section_header("DEMO 5: Batch Captioning")
    print("\n  Processing all sample images with the same candidate captions...\n")

    try:
        batch_captions = [
            "a photo of a dog",
            "a photo of a cat",
            "a photo of a busy city street",
            "a photo of a mountain landscape",
            "a photo of food on a plate",
        ]

        batch_results = captioner.batch_caption(
            list(sample_images.values()), batch_captions
        )

        for i, result in enumerate(batch_results):
            label = list(sample_images.keys())[i]
            print(f"\n  Image: {label}")
            if result["status"] == "success":
                print(f"    Best Caption : {result['best_caption']}")
                print(f"    Confidence   : {result['confidence']:.4f}")
            else:
                print(f"    Error: {result['error']}")
    except Exception as e:
        print(f"  Error in Demo 5: {e}")

    # ===== SUMMARY =====
    print_section_header("SUMMARY: CLIP Multimodal Capabilities")
    print("""
  CLIP (Contrastive Language-Image Pre-Training) bridges vision and language
  by learning a shared embedding space for images and text.

  Key Capabilities Demonstrated:
    1. Zero-shot Image Captioning   — Rank candidate captions for an image
    2. Zero-shot Classification     — Categorize images without training data
    3. Image-Text Similarity        — Compute cosine similarity between modalities
    4. Cross-modal Retrieval        — Find the best image for a given text query
    5. Batch Processing             — Efficiently process multiple images

  Architecture:
    • Image Encoder  : Vision Transformer (ViT-B/32)
    • Text Encoder   : Transformer-based text encoder
    • Training       : Contrastive learning on 400M image-text pairs
    • Embedding Dim  : 512-dimensional shared space

  Advantages:
    ✓ No task-specific fine-tuning required (zero-shot)
    ✓ Understands diverse visual concepts via natural language
    ✓ Lightweight and fast inference

  Limitations:
    ✗ Cannot generate novel captions (ranks from candidates only)
    ✗ Performance depends on quality of candidate captions
    ✗ May struggle with fine-grained visual details
""")
    print("=" * 70)
    print("  Demo complete! Thank you for exploring CLIP-based image captioning.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

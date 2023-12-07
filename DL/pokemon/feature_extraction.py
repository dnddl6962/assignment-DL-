# feature_extraction.py

import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

class ImageSimilarityCalculator:
    def __init__(self, image_folder, top_k=6, num_cols=3):
        self.image_folder = image_folder
        self.top_k = top_k
        self.num_cols = num_cols
        self.model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.features_dict = self._extract_all_features()

    def _extract_features(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = self.model.predict(img)
        return features.flatten()

    def _extract_all_features(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        features_dict = {}
        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            features = self._extract_features(image_path)
            features_dict[image_file] = features
        return features_dict

    def calculate_cosine_similarity(self, selected_image_path):
        selected_image_features = self._extract_features(selected_image_path)
        similarities = {}
        for image_file, features in self.features_dict.items():
            similarity = cosine_similarity([selected_image_features], [features])[0][0]
            similarities[image_file] = similarity
        return similarities

    def plot_top_similar_images(self, sorted_similarities):
        num_rows = (self.top_k + self.num_cols - 1) // self.num_cols
        fig, axes = plt.subplots(num_rows, self.num_cols, figsize=(15, 5))

        for i in range(self.top_k):
            similar_image_path = os.path.join(self.image_folder, sorted_similarities[i][0])
            similarity_score = sorted_similarities[i][1]

            # 이미지를 Matplotlib을 사용하여 표시
            row = i // self.num_cols
            col = i % self.num_cols
            axes[row, col].imshow(imread(similar_image_path))
            axes[row, col].set_title(f"Similarity: {similarity_score:.2f}")
            axes[row, col].axis("off")

        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 서브플롯 간의 간격 조정
        plt.show()

    def find_and_plot_similar_images(self, selected_image_path):
        similarities = self.calculate_cosine_similarity(selected_image_path)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        self.plot_top_similar_images(sorted_similarities)

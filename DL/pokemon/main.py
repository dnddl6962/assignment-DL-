from feature_extraction import ImageSimilarityCalculator

# 이미지 경로
image_folder = "./images"

# ImageSimilarityCalculator 클래스 인스턴스 생성
image_similarity_calculator = ImageSimilarityCalculator(image_folder)

# 사용자가 선택한 이미지에 대한 유사도 계산 및 플로팅
selected_image_path = "./images/zeraora.jpg"
image_similarity_calculator.find_and_plot_similar_images(selected_image_path)

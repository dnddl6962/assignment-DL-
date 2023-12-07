# assignment-DL-
딥러닝 과제 제출용

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ 영화 추천 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

환경
python --version == 3.9.15
pd.__version__ == '2.0.3'
sklearn.__version__ == '1.2.2'
tf.__version__ == '2.8.0'
deepctr.__version__ == '0.9.3'

구성
movielens.csv --> 기존 데이터셋
module.py --> 모듈
main.py --> 실행 파일
predictions.csv --> 결과물(유저아이디, 영화이름, 유저가 영화를 볼 확률)
ossme_movie.ipynb --> 분기 전 주피터 노트북 파일

실행 방법 
main.py의 og_path는 기존 데이터셋의 경로, new_path는 결과물을 받을 경로 입니다.
module.py안의 클래스의 이름이 movie라서 변수에 클래스를 받는데 og_path가 인자입니다.
그리고 결과물을 받는 함수는 to_csv로 클래스 안에 있습니다. to_csv는 new_path를 인자로 받아 결과물을 생성후 new_path에 적힌 경로로 저장합니다. 


\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ 유사 포켓몬 추출 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


환경
python --version == 3.11.6
cv2.__version__ == '4.8.1'
sklearn.__version__ == '1.3.2'
tf.__version__ == '2.15.0'
np.__version__ == '1.26.2'
matplotlib.__version__ == '3.8.2'  

구성
images(디렉토리) --> 포켓몬 이미지 데이터 셋
feature_extraction.py --> 모듈
main.py --> 실행 파일
pokemon.csv --> 포켓몬 정보 csv파일(사용안함)
pika.ipynb --> 분기 전 주피터 노트북 파일

실행 방법 
feature_extraction.py의 ImageSimilarityCalculator 클래스를 가져와서 사용합니다.
필수 인자는 포켓몬 이미지 데이터 셋의 경로입니다.
main.py에서 변수에 포켓몬 이미지 데이터 셋의 경로를 입력하고 클래스를 불러와서 기입합니다.
ex)
# 이미지 경로
image_folder = "./images"
# ImageSimilarityCalculator 클래스 인스턴스 생성
image_similarity_calculator = ImageSimilarityCalculator(image_folder)

그리고 선택한 포켓몬과 유사한 포켓몬들을 보려면 포켓몬 이미지의 경로를 변수에 입력하고 클래스안의 메소드 find_and_plot_similar_images를 불러와 기입합니다.
ex)
# 사용자가 선택한 이미지에 대한 유사도 계산 및 플로팅
selected_image_path = "./images/zeraora.jpg"
image_similarity_calculator.find_and_plot_similar_images(selected_image_path)

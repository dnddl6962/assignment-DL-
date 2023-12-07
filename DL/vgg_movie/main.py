from module import movie

og_path = "/Users/bearjang/Desktop/bigdata_5th/DL/vgg_movie/movielens.csv" # 기존 영화 csv경로
new_path = "/Users/bearjang/Desktop/bigdata_5th/DL/vgg_movie/predictions.csv" # 새로 받을 csv경로

movie = movie(og_path)
movie.to_csv(new_path)

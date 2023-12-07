import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat,get_feature_names
from deepctr.models import DeepFM




class movie:
    def __init__(self, og_folder):
        self.og_folder = og_folder
        self.sparse_features = ['userId', 'title', 'genres', 'tag', 'rating']
        self.target = ['target']
        self.field_info = dict(userId = 'user', title = 'context', genres = 'context', tag ='context', rating = 'context')
        self.data_preprocessing()
        self.deepfm()


    #데이터 전처리 부분    
    def data_preprocessing(self):
        self.data = pd.read_csv(self.og_folder)
        for feat in self.sparse_features:
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])

        self.fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=self.data[feat].nunique(), embedding_dim=4) for feat in self.sparse_features]
        self.dnn_feature_columns = self.fixlen_feature_columns
        self.linear_feature_columns = self.fixlen_feature_columns

        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)

        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=2020)
        self.train_model_input = {name: self.train[name] for name in self.feature_names}
        self.test_model_input = {name: self.test[name] for name in self.feature_names}


    #모델 돌리기 ~~~~
    def deepfm(self):
        self.model = DeepFM(self.linear_feature_columns, self.dnn_feature_columns, task='binary')
        self.model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

        self.history = self.model.fit(self.train_model_input, self.train[self.target].values,
                        batch_size=256, epochs=20, verbose=2, validation_split=0.2, )
        self.pred_ans = self.model.predict(self.test_model_input, batch_size=256)
        print("test LogLoss", round(log_loss(self.test[self.target].values, self.pred_ans), 4))
        print("test AUC", round(roc_auc_score(self.test[self.target].values, self.pred_ans), 4))


    #csv저장(디코딩 합쳐서)
    def to_csv(self, new_folder):
        self.new_folder = new_folder
        self.predictions = pd.DataFrame({
        'userId': self.test['userId']+1,
        'title': self.test['title'],
        'prediction': self.pred_ans.flatten()})
    
        self.data = pd.read_csv(self.og_folder)
        self.lbe = LabelEncoder()
        self.lbe.fit(self.data['title'])
        self.predictions['title'] = self.lbe.inverse_transform(self.predictions['title'])
        self.predictions.to_csv(self.new_folder, index=False)
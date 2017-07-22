import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

data_file_name = 'Datasets\data_mi_imp.pkl'
# unpickle data set
with open(data_file_name, 'rb') as f:
    pkl_data = pickle.load(f)

df = pkl_data['test_data']
ids = pkl_data['test_ID']

############### NORMALIZATION #################
#rs = RobustScaler()
#rs = rs.fit(df)
#df.iloc[:] = rs.transform(df)
###############################################

X = np.array(df.values)

with open('learning_model_ams_dt_bagging_20.pkl', 'rb') as f:
    cla_pkl = pickle.load(f)

cla = cla_pkl['model']
print(cla.get_params)

Y_pred = cla.predict(X)
scores = cla.predict_proba(X)
scores = np.max(scores, axis=1)
s_filter = Y_pred == 1

scores[s_filter] += 1
RankOrder = np.argsort(scores)

Y_pred_str = np.array(['s' if pred == 1 else 'b' for pred in Y_pred])

submission = pd.DataFrame()
submission['EventId'] = ids[RankOrder]
submission['RankOrder'] = np.arange(1, len(scores) + 1)
submission['Class'] = Y_pred_str[RankOrder]

submission.to_csv("Submission\Hajj_submission_dt.csv", index=False)
print(submission)

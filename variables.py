import os
classifier_idx = 1
cutoff = 0.90
batch_size = 128
num_epoches = 20
seed = 42
depth = 80
n_neighbors = 3
model_name = 'Support Vector Machine'
C = 1.0
model_id = 'C[{}]'.format(C)
pred_dict = {True : 'positive' , False : 'negative'}
train_csv = 'train_reviews.csv'
test_csv  = 'test_reviews.csv'
raw_test_csv = 'user_test_reviews.csv'
raw_train_csv = 'user_reviews.csv'
pkl_filename = 'Models/{} {} {}.pkl'.format(model_name, classifier_idx, model_id)
csv_path = 'Results/{} {} {}.csv'.format(model_name, classifier_idx, model_id)
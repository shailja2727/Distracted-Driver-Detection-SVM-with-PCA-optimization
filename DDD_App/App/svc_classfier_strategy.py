import pickle
import numpy as np
with open('model_data.pkl', 'rb') as f:
    clf = pickle.load(f)
print("Strategy used by SVC:")
print(clf['model'].decision_function_shape)

print('w = ',clf['model'].coef_.shape)
print('b = ',clf['model'].intercept_.shape)
print('Indices of support vectors = ', clf['model'].support_.shape)
print('Support vectors = ', clf['model'].support_vectors_.shape)
print('Number of support vectors for each class = ', clf['model'].n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf['model'].dual_coef_))
import lasso
import numpy as np
import csv


Xy = []
with open('winequality-red.csv') as fp:
    for row in csv.reader(fp, delimiter=';'):
        Xy.append(row)

Xy = np.array(Xy[1:], dtype=np.float64)

np.random.seed(0)
np.random.shuffle(Xy)

train_x = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_x = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]


results = []
for lambda_ in [1., 0.1, 0.01]:
    model = lasso.Lasso(lambda_)
    model.fit(train_x, train_y)
    y = model.predict(test_x)
    print(f'---result by lambda = {lambda_} ---')
    print('coefficients')
    print(model.w_)
    mse = ((y-test_y)**2).mean()
    print(f'MSE is {mse: .3f}, lambda is {lambda_}')
    print('''
    
    ''')
    results.append([mse, lambda_])

print(f'the best MSE and lambda is {max(results)}')



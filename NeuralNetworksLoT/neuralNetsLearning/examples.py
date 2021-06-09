from neural_network import CategoryLearner
from itertools import product
import numpy as np
from numpy.random import randint
from pprint import pprint
import matplotlib.pyplot as plt

n_properties = 4
reps = 10

objects = np.array(list(product([0,1],repeat=n_properties)))
print(objects)
category = randint(0,2**len(objects))
print(category)
for a in range(8):
    learner = CategoryLearner(category, objects, 4, 500, reps)
    learner.train_on_category()
    # data = learner.learning_curves[0].mean(axis=1)
    print(np.array(learner.learning_curves).shape)
    for data in learner.learning_curves:
        ys = data.mean(axis=1)
        xs = np.arange(len(data))
        plt.scatter(
            xs, ys,
            s=0.1, 
            c=[a]*len(data)
        )
plt.show()

# xs = np.linspace(-10,10,100)
# ys = logistic_function(xs, 2, -5, 0.4)
# plt.plot(xs,ys)
# plt.show()

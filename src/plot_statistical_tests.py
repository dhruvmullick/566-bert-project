import numpy as np
import matplotlib.pyplot as plt

'''
Plot results of statistical tests
'''

values = []
file = open("data/multi_layer_stat_test", "r")
for line in file.readlines():
    values += [float(line.strip('\n'))]
file.close()

fig, ax = plt.subplots(1, 1)
ax.hist(values, bins=20)
ax.set_title("Multi Layer Permutation Scores")
ax.set_xlabel('MSE Scores')
ax.set_ylabel('No. of Outcomes')
plt.show()

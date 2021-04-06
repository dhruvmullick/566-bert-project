import numpy as np
import matplotlib.pyplot as plt

'''
Plot results of statistical tests
'''

values = []
file = open("data/single_layer_stat_test", "r")
for line in file.readlines():
    values += [float(line.strip('\n'))]
file.close()

fig, ax = plt.subplots(1, 1)
ax.hist(values, bins=20, label='Permutation Scores')
ax.set_title("Single Layer Permutation Scores")
ax.set_xlabel('MSE Scores')
ax.set_ylabel('No. of Outcomes')
# plt.axvline(x=0.267662, color='red', linestyle='--', label='Observed Score')
plt.axvline(x=0.164762, color='red', linestyle='--', label='Observed Score')
plt.legend()
plt.show()

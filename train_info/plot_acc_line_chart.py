import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

f = open('loss_acc_per_100_2.txt', 'r').readlines()
f = [ float(i.split(' ')[-1][:-2]) for i in f if 'poch' not in i]

x = range(len(f))

plt.plot(x, f)
plt.show()
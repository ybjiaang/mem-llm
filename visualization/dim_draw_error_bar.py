import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.font_manager import FontProperties

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

font = FontProperties()
font.set_family('serif')
font.set_size('12')

# # Sample data
# x = [16, 32, 64, 128, 256, 512]
# data = pd.DataFrame({
#     'x': [16, 32, 64, 128, 256, 512],
#     'y': np.sin(np.arange(0, 6, 1)),
#     'y_err': 0.1 * np.sqrt(np.arange(0, 6, 1))  # Sample error bars
# })

# # Plot with error bars
# sns.lineplot(data=data, x='x', y='y')
# plt.errorbar(data['x'], data['y'], yerr=data['y_err'], fmt='o', color='red', capsize=5)

M = 8
x = np.array([16, 32, 64, 128, 256, 512])
if M == 5:
    y_freeze = np.array([0.6595703125, 0.989453125, 0.998046875, 0.9998046875, 0.9994140625, 0.985546875])
    y_freeze_err = np.array([0.021174973827647638, 0.001987010737243731, 0.0006905339660024879, 0.00019531250000000004, 0.0005859374999999999, 0.0050875063584496505])

    y_no_freeze = np.array([0.9982421875, 0.9955078125, 0.9998046875, 1.0, 0.9982421875, 0.983984375])
    y_no_freeze_err = np.array([0.00019531249999999998, 0.0010517900013934578, 0.00019531250000000004, 0.0, 0.0005694288959809863, 0.006227069752706976])

if M == 6:
    y_freeze = np.array([0.1974609375, 0.7330078125, 0.9904296875, 0.997265625, 0.99921875, 0.9857421875])
    y_freeze_err = np.array([0.0035129112790244035, 0.014702374657062735, 0.0014285877771804485, 0.0004784159653873395, 0.0005694288959809863, 0.007767202666073506])

    y_no_freeze = np.array([0.9927734375, 0.997265625, 0.9982421875, 0.9986328125, 0.9990234375, 0.9783203125])
    y_no_freeze_err = np.array([0.001401630868594465, 0.0010874539771152385, 0.0006477782793662889, 0.0006623369124145769, 0.0003088161777508183, 0.017325110514675026])


if M == 7:
    y_freeze = np.array([0.05859375, 0.237890625, 0.807421875, 0.9818359375, 0.9982421875, 0.9693359375])
    y_freeze_err = np.array([0.0060357566954540795, 0.008249496907008937, 0.008660077842732563, 0.0052734374999999995, 0.00019531250000000004, 0.013342145550862147])

    y_no_freeze = np.array([0.9736328125, 0.9951171875, 0.9958984375, 0.9990234375, 0.9984375, 0.983203125])
    y_no_freeze_err = np.array([0.006737573558851695, 0.0009264485332524547, 0.0010426834230499327, 0.0005348853100636387, 0.000662336912414577, 0.005405606446522674])


if M == 8:
    y_freeze = np.array([0.01953125, 0.059765625, 0.2576171875, 0.7998046875, 0.9755859375, 0.96875])
    y_freeze_err = np.array([0.001512884119612272, 0.0056218117049313485, 0.005553541878169113, 0.004085254035810916, 0.0016341016143243661, 0.009077304717673634])

    y_no_freeze = np.array([0.8857421875, 0.990234375, 0.986328125, 0.996484375, 0.9935546875, 0.9828125])
    y_no_freeze_err = np.array([0.03525647605765366, 0.0018269811458857135, 0.002290242070226284, 0.0011799849583588448, 0.0013671875, 0.002667285211669465])



# Plot with error band
sns.lineplot(x=x, y=y_freeze, label='Frozen Embeddings')
plt.fill_between(x, y_freeze - y_freeze_err, y_freeze + y_freeze_err, alpha=0.2)

sns.lineplot(x=x, y=y_no_freeze, label='Trained Embeddings')
plt.fill_between(x, y_no_freeze - y_no_freeze_err, y_no_freeze + y_no_freeze_err, alpha=0.2)


plt.xlabel('Dimensions', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
# plt.title('M = ' + str(M), fontsize=20)
plt.grid(True)
plt.xscale('log')
plt.xticks([16, 32, 64, 128, 256, 512], ['16', '32', '64', '128', '256', '512'])
plt.axvline(x=2**M, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("dims_plot/dim_n"+str(M)+".png")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

seq_len = 128

if seq_len == 128:
    # Sample data
    data = {
        'Category': ['5', '6', '7', '8'],
        'Identity Wv': [0.817578125, 0.865234375, 0.8869140625, 0.9029296875],
        'Trained Wv': [0.98515625, 0.97666015625, 0.96171875, 0.9431640625],
        
    }

    frozen_stderr = np.array([0.004015402069038873, 0.0022458578529505648, 0.0029108188679828327, 0.0026131031563007127])
    training_stderr= np.array([0.001332649055368406, 0.0015019885410233497, 0.0017638303625187588, 0.002273058626314595])

if seq_len == 64:
    data = {
        'Category': ['5', '6', '7', '8'],
        'Identity Wv': [0.5705078125, 0.6244140625, 0.6595703125, 0.65791015625],
        'Trained Wv': [0.8912109375, 0.8697265625, 0.8197265625, 0.7587890625],
        
    }

    frozen_stderr = np.array([0.007608836233921414, 0.006460091956461416, 0.003527660725037517, 0.004303897552102906])
    training_stderr= np.array([0.004136036503715779, 0.00406105994108091, 0.0033394058862583746, 0.00505344228393881])


# Convert data to a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='Category', var_name='Group', value_name='Value')

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Value', hue='Group', data=df_melted)
plt.errorbar(x=[index - 0.2 for index in range(len(df_melted['Category'].unique()))], y=df['Identity Wv'], yerr=frozen_stderr, fmt='none', color='black', capsize=5)
plt.errorbar(x=[index + 0.2 for index in range(len(df_melted['Category'].unique()))], y=df['Trained Wv'], yerr=training_stderr, fmt='none', color='black', capsize=5)
plt.xlabel('Number of Latent Concepts', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# Adjust legend
plt.legend(title='', loc='lower left')
plt.savefig("Wv_bar_l" + str(seq_len) + ".png")

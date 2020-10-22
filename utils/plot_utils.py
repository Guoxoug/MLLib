import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
sns.set(color_codes=True)

def plot_reliability_curve(probs_, targets_, file_path=None, n_bins=15):
    plt.plot([0, 1], [0, 1], linestyle='--')
    fop, mpv = calibration_curve((targets_==np.argmax(probs_,axis=1)), np.max(probs_, axis=1), n_bins=n_bins)
    plt.plot(mpv, fop, marker='v')
    plt.ylabel('Fraction of corrects')
    plt.xlabel('Max Probability')
    plt.title('Top-label Reliability Curve')
    plt.legend()
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
    else:
        plt.show()
    #plt.close()

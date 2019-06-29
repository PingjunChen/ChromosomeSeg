# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('chromosome_seg_cmp.pdf')

unet_sum_nonoverlap_dice = [0.803, 0.910, 0.943, 0.948, 0.952, 0.952, 0.949, 0.949, 0.952, 0.953]
unet_random_nonoverlap_dice = [0.807, 0.932, 0.947, 0.950, 0.947, 0.953, 0.953, 0.956, 0.954, 0.953]

unet_sum_overlap_dice = [0.351, 0.793, 0.818, 0.835, 0.863, 0.859, 0.856, 0.849, 0.836, 0.830]
unet_random_overlap_dice = [0.282, 0.832, 0.849, 0.852, 0.850, 0.855, 0.854, 0.864, 0.858, 0.859]

psp_sum_nonoverlap_dice = [0.766, 0.841, 0.847, 0.852, 0.857, 0.869, 0.874, 0.873, 0.870, 0.877]
psp_random_nonoverlap_dice = [0.751, 0.858, 0.865, 0.873, 0.864, 0.861, 0.864, 0.873, 0.867, 0.858]

psp_sum_overlap_dice = [0.525, 0.764, 0.772, 0.786, 0.804, 0.804, 0.832, 0.832, 0.847, 0.845]
psp_random_overlap_dice = [0.476, 0.825, 0.847, 0.842, 0.835, 0.840, 0.828, 0.832, 0.836, 0.832]

# common
x_inds = np.arange(len(unet_sum_nonoverlap_dice))
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(8, 20), dpi=300)

# ax1
ax1.plot(x_inds, unet_sum_nonoverlap_dice, '-o', label="Sum")
ax1.plot(x_inds, unet_random_nonoverlap_dice, '-s', label="Random")
ax1.scatter(1, 0.822, c="r")
ax1.text(0.60, 0.81, r"$\beta_1=1,\beta_2=0$", fontsize=6)

ax1.set_ylabel("Dice coefficient")
# ax1.set_xticks(x_inds)
ax1.set_xticks([])
ax1.legend()
ax1.set_title("UNet on Non-overlapped Region", fontsize=10)

# ax2
ax2.plot(x_inds, unet_sum_overlap_dice, '-o', label="Sum")
ax2.plot(x_inds, unet_random_overlap_dice, '-s', label="Random")
ax2.scatter(1, 0.590, c="r")
ax2.text(0.60, 0.54, r"$\beta_1=1,\beta_2=0$", fontsize=6)
ax2.set_xticks([])
ax2.legend()
ax2.set_title("UNet on Overlapped Region", fontsize=10)


# ax3
ax3.plot(x_inds, psp_sum_nonoverlap_dice, '-o', label="Sum")
ax3.plot(x_inds, psp_random_nonoverlap_dice, '-s', label="Random")
ax3.scatter(1, 0.618, c="r")
ax3.text(0.60, 0.63, r"$\beta_1=1,\beta_2=0$", fontsize=6)

ax3.set_xlabel(r'$\beta_{1}, \beta_{2}=1$')
ax3.set_ylabel("Dice coefficient")
ax3.set_xticks(x_inds)
plt.ylim(0.60, 1.0)
ax3.legend()
ax3.set_title("PSPNet on Non-overlapped Region", fontsize=10)


# ax4
ax4.plot(x_inds, psp_sum_overlap_dice, '-o', label="Sum")
ax4.plot(x_inds, psp_random_overlap_dice, '-s', label="Random")
ax4.scatter(1, 0.578, c="r")
ax4.text(0.60, 0.60, r"$\beta_1=1,\beta_2=0$", fontsize=6)

ax4.set_xlabel(r'$\beta_{1}, \beta_{2}=1$')
ax4.set_xticks(x_inds)
ax4.legend()
ax4.set_title("PSPNet on Overlapped Region", fontsize=10)


# plt.tight_layout()
plt.show()

# pp.savefig(fig)
# pp.close()

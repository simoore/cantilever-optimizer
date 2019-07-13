import numpy as np
import matplotlib.pyplot as plt


front1 = np.load('solutions-off-resonance/prelim-1a-front.npy')
front2 = np.load('solutions-off-resonance/prelim-1b-front.npy')
front3 = np.load('solutions-off-resonance/prelim-1c-front.npy')


fig, ax = plt.subplots()
ax.scatter(-front1[:,0]*1e-3, front1[:,1], c='b', label='Low Detail')
ax.scatter(-front2[:,0]*1e-3, front2[:,1], c='r', label='High Detail')
ax.scatter(-front3[:,0]*1e-3, front3[:,1], c='g', label='Rectangular')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Stiffness (N/m)')
ax.set_ylim(0, 6000)
ax.set_xlim(0, 10000)
ax.legend()


fig, ax = plt.subplots()
ax.scatter(-front1[:,0]*1e-3, front1[:,1], c='b', label='Low Detail')
ax.scatter(-front2[:,0]*1e-3, front2[:,1], c='r', label='High Detail')
ax.scatter(-front3[:,0]*1e-3, front3[:,1], c='g', label='Rectangular')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Stiffness (N/m)')
ax.set_ylim(0, 50)
ax.set_xlim(0, 100)
ax.legend()



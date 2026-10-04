# -*- coding: utf-8 -*-
"""plotting.py -- all plots, using the files written by analysis.py."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyccl as ccl

data_sorted = pd.read_csv("outputs/data_sorted.csv")
covar_sorted = np.load("outputs/covar_sorted.npy")
fcovar = np.load("outputs/fcovar.npy")
z_sorted = data_sorted['z'].values

normal_w_priors = np.load("outputs/normal_w_priors.npy")
normal_wout_priors = np.load("outputs/normal_wout_priors.npy")
splined_w_priors = np.load("outputs/splined_w_priors.npy")
splined_wout_priors = np.load("outputs/splined_wout_priors.npy")

from matplotlib.colors import LogNorm
plt.figure(figsize=(10, 8))
z_labels = [f"{val:.2f}" for val in data_sorted['z']]

plt.imshow(
    fcovar,
    cmap="binary",
    norm=LogNorm(vmin=fcovar[fcovar>0].min(),vmax=fcovar.max()))

plt.xticks(ticks=np.arange(len(z_labels)), labels=z_labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(z_labels)), labels=z_labels, fontsize=8)

plt.title("Covariance Matrix indexed by Redshift (z)")
plt.colorbar(label='Covariance Value')
plt.xlabel("z")
plt.ylabel("z")
plt.show()

dcovar = fcovar-covar_sorted

plt.imshow(
    covar_sorted,
    cmap="binary",
    norm=LogNorm(vmin=covar_sorted[covar_sorted>0].min(), vmax=covar_sorted.max())
)

plt.xticks(ticks=np.arange(len(z_labels)), labels=z_labels, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(z_labels)), labels=z_labels, fontsize=8)

plt.title("Covariance Matrix indexed by Redshift (z)")
plt.colorbar(label='Covariance Value')
plt.xlabel("z")
plt.ylabel("z")
plt.show()

# Calculate correlation matrix: corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
ddiag = np.sqrt(np.diag(fcovar))
corr_matrix = fcovar / ddiag[:, None] / ddiag[None, :]

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
z_labels_sorted = [f"{val:.2f}" for val in z_sorted]

sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="RdBu_r",
    center=0,
    xticklabels=z_labels_sorted,
    yticklabels=z_labels_sorted,
    cbar_kws={'label': 'Correlation Coefficient'}
)

plt.title("Correlation Matrix (Sorted by Redshift z)")
plt.xlabel("z")
plt.ylabel("z")
plt.show()

def get_H_bounds(H0, dH0):
    cosmo = ccl.Cosmology(Omega_c=0.2422, Omega_b=0.045, h=H0/100, sigma8=0.81, n_s=0.96)

    z_arr = np.linspace(0.8*z_sorted.min(), 1.05*z_sorted.max(), 128)
    a_arr = 1 / (1 + z_arr)
    H_model = 100 * cosmo["h"] * cosmo.h_over_h0(a_arr)

    cosmo_min = ccl.Cosmology(Omega_c=0.2422, Omega_b=0.045, h=(H0-dH0)/100, sigma8=0.81, n_s=0.96)
    H_model_min = 100 * cosmo_min["h"] * cosmo_min.h_over_h0(a_arr)

    cosmo_max = ccl.Cosmology(Omega_c=0.2422, Omega_b=0.045, h=(H0+dH0)/100, sigma8=0.81, n_s=0.96)
    H_model_max = 100 * cosmo_max["h"] * cosmo_max.h_over_h0(a_arr)
    return z_arr, H_model, H_model_min, H_model_max


z_arr, H_model, H_model_min, H_model_max = get_H_bounds(70.5, 1.1)  # informed prior
_, H_model_flat, H_model_min_flat, H_model_max_flat = get_H_bounds(66.5, 4.2)  # flat prior
z_arr, H_model3, H_min3, H_max3 = get_H_bounds(71.9, 4.2) #splined flat
_, H_model4, H_min4, H_max4 = get_H_bounds(70.8,1.1)#splined informed

# Extract diagonal errors from the sorted covariance matrix
errors = np.sqrt(np.diag(covar_sorted))

plt.figure(figsize=(12, 7))

# Use seaborn to manage color mapping by 'reference'
sns.scatterplot(
    data=data_sorted,
    x='z',
    y='Hz',
    hue='reference',
    style='reference',
    s=100,
    zorder=3
)

# Add error bars manually
plt.errorbar(
    data_sorted['z'],
    data_sorted['Hz'],
    yerr=errors,
    fmt='none',
    ecolor='gray',
    alpha=0.5,
    zorder=2
)

plt.fill_between(z_arr, y1=H_model_min_flat, y2=H_model_max_flat, color="skyblue", alpha=0.5, label=r"$\Delta H(z)$ (flat, not splined)")
plt.plot(z_arr, H_model_flat, c="navy", ls="--", label=r"$\Lambda$ CDM fit, flat, not splined")

plt.fill_between(z_arr, y1=H_model_min, y2=H_model_max, color="pink", alpha=0.5, label=r"$\Delta H(z)$ (informed, not splined)")
plt.plot(z_arr, H_model, c="crimson", ls="--", label=r"$\Lambda$ CDM fit, informed, not splined")

plt.plot(z_arr, H_model3, color='darkgreen', linestyle='--', label=r"$\Lambda$ CDM fit, flat, splined")
plt.fill_between(z_arr, H_min3, H_max3, color='lightgreen', alpha=0.5, label=r"$\Delta H(z)$ (flat, splined)")

plt.plot(z_arr, H_model4, color='purple', linestyle='-.', label=r"$\Lambda$ CDM fit, informed, splined")
plt.fill_between(z_arr, H_min4, H_max4, color='plum', alpha=0.5, label=r"$\Delta H(z)$ (informed, splined)")

plt.xlabel('Redshift (z)', fontsize=12)
plt.ylabel('H(z) [km/s/Mpc]', fontsize=12)
plt.legend(loc='upper left', ncol=3, fontsize=9, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(x=normal_w_priors[:,0], ax=axes[0], color="crimson", label="Untreated with informed priors", linewidth=2)
sns.kdeplot(x=normal_wout_priors[:,0], ax=axes[0], color="navy", label="Untreated with flat priors", linewidth=2)
sns.kdeplot(x=splined_w_priors[:,0], ax=axes[0], color="purple", label="Splined with informed priors", linewidth=2)
sns.kdeplot(x=splined_wout_priors[:,0], ax=axes[0], color="darkgreen", label="Splined with flat priors", linewidth=2)

sns.kdeplot(x=normal_w_priors[:,1], ax=axes[1], color="crimson", label="Untreated with informed priors", linewidth=2)
sns.kdeplot(x=normal_wout_priors[:,1], ax=axes[1], color="navy", label="Untreated with flat priors", linewidth=2)
sns.kdeplot(x=splined_w_priors[:,1], ax=axes[1], color="purple", label="Splined with informed priors", linewidth=2)
sns.kdeplot(x=splined_wout_priors[:,1], ax=axes[1], color="darkgreen", label="Splined with flat priors", linewidth=2)

axes[0].set_xlabel("$H_0$")
axes[1].set_xlabel("$\Omega_m$")
axes[0].set_ylabel("Density")

axes[0].legend(loc="upper left", frameon=True)
axes[1].legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.show()

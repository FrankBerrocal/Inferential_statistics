"""
Fundatec / Procomer
Programa técnico en aseguramiento de la calidad

Mayo 2026

Descriptive statistical analysis of random data
Frank Berrocal  

Goal: implementation of position, spare, and structure metrics 
using primitives, not libraries. 
"""


import math
import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import skew, kurtosis

df_polars = pl.read_csv('/Users/frankberrocalazofeifa/Documents/GitHub/Inferential_statistics/Synthetic_Financial_datasets_log.csv',
                        columns=['amount'])
vecteur_haute_vitesse = df_polars['amount'].to_numpy()

# Summary Statistics
# Count elements in set
# Sum of values
# Minimum value
# Maximum value

def nan_element_count(n_array):
    valid_count = vecteur_haute_vitesse.size - \
    np.sum(~np.isnan(vecteur_haute_vitesse))
    return valid_count

def mode_calculation(n_array):
    mode_result = stats.mode(vecteur_haute_vitesse, keepdims=False)
    le_mode = mode_result.mode
    frequence_du_mode = mode_result.count
    return le_mode, frequence_du_mode


def Coefficient_variation(n_array):
    cv = 0
    cv = ((np.std(n_array)/np.mean(n_array))*100)
    return cv


def percentileCalculation(n_array):
    quantile25, quantile50, quantile75 = np.percentile(
        n_array, [25, 50, 75])
    return quantile25, quantile50, quantile75


# 1. Quantify Asymmetry
# bias=False applies the sample correction factor (Bessel's correction logic) for finite N
# nan_policy='omit' ensures the C-loop does not crash if corrupted memory (NaN) is encountered

def skewness_calculation(n_array):
    asymmetry_factor = skew(n_array,
                            bias=False, nan_policy='omit')
    return asymmetry_factor

# 2. Quantify Tail Weight / Outlier Propensity
# fisher=True sets the Normal baseline to 0.0


def kurtosis_calculation(n_array):
    tail_extremity = kurtosis(n_array,
                              fisher=True, bias=False, nan_policy='omit')
    return tail_extremity


# 3.  Fisher's Coefficient
def fisher_calculation(n_array):
    fisher_coefficient = math.pow(
        np.mean(n_array), 3) / math.pow(np.std(n_array), 3)
    return fisher_coefficient

# 4.  Pearson's Coefficient


def pearson_calculation(n_array):
    pearson_coefficient = (np.mean(n_array) -
                           mode_calculation(n_array)[0]) / np.std(n_array)
    return pearson_coefficient


def robust_scaling_algorithm(vecteur_haut_vitesse):
    # Raw data to predictive-usable data
    # 1. Spatial extraction of percentiles
    percentiles = np.nanpercentile(vecteur_haute_vitesse, [25, 50, 75])

    q1 = percentiles[0]
    mediane = percentiles[1]
    q3 = percentiles[2]

    iqr = q3 - q1

    # IQR cannot be null (50% up above 50% below)
    if iqr == 0:
        iqr = 1e-9  # epsilon material

    # 2. Subtract the median to n-array, then divide by IQR
    vecteur_compresse = (vecteur_haute_vitesse - mediane) / iqr
    return vecteur_compresse


# Output report
# Measure of base information
print(f"\n\nMeasures of base information\n")
print(f"Total elements in set: {vecteur_haute_vitesse.size}")
print(f"Total sum of values: {vecteur_haute_vitesse.sum()}")
print(f"Minimum value in observations: {vecteur_haute_vitesse.min()}")
print(f"Maximum value in observations: {vecteur_haute_vitesse.max()}")
print(f"NaN values: {nan_element_count(vecteur_haute_vitesse)}")
print(f"NaN values: {nan_element_count(vecteur_haute_vitesse)}")

# Measures of central tendency
print(f"\n\nMeasures of central tendency\n")
print(f"Median: {np.median(vecteur_haute_vitesse)}")
print(f"Mean: {np.mean(vecteur_haute_vitesse)}")
print(f"Mode: {mode_calculation(vecteur_haute_vitesse)[0]} with frequency {mode_calculation(vecteur_haute_vitesse)[1]}")

# Measures of variablity
print(f"\n\nMeasures of variability\n")
print(f"Range: {np.ptp(vecteur_haute_vitesse)}")  # difference is too low
print(f"Variance: {np.var(vecteur_haute_vitesse)}")
# same number, no NaN values found
print(f"NaN Variance: {np.nanvar(vecteur_haute_vitesse)}")
print(f"Standard Deviation: {np.std(vecteur_haute_vitesse)}")
# same number, no NaN values found
print(f"NaN STD: {np.nanstd(vecteur_haute_vitesse)}")
print(
    f"Coefficient of variation: {Coefficient_variation(vecteur_haute_vitesse)}%")
print(f"Quantile 25%: {percentileCalculation(vecteur_haute_vitesse)[0]}")
print(f"Quantile 50%: {percentileCalculation(vecteur_haute_vitesse)[1]}")
print(f"Quantile 75%: {percentileCalculation(vecteur_haute_vitesse)[2]}")

# Measures of geometry
print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(vecteur_haute_vitesse)}")
print(f"Pearson's Coefficient: {pearson_calculation(vecteur_haute_vitesse)}")
print(f"Skewness: {skewness_calculation(vecteur_haute_vitesse)}")
# severe leptokurtic behavior (extreme positive kurtosis)
print(f"Kurtosis: {kurtosis_calculation(vecteur_haute_vitesse)}")

# Data model compression
print(f"\n\nData Model Compression **************\n")
print(f"Robust Scaling\n")
vecteur_compresse = robust_scaling_algorithm(vecteur_haute_vitesse)
# Measures of variablity
print(f"Measures of variability\n")
print(f"Range: {np.ptp(vecteur_compresse)}")  # now a range is being generated
print(f"Variance: {np.var(vecteur_compresse)}")
print(f"Standard Deviation: {np.std(vecteur_compresse)}")


# Measures of geometry
print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(vecteur_compresse)}")
print(f"Pearson's Coefficient: {pearson_calculation(vecteur_compresse)}")
print(f"Skewness: {skewness_calculation(vecteur_compresse)}")
# severe leptokurtic behavior (extreme positive kurtosis)
print(f"Kurtosis: {kurtosis_calculation(vecteur_compresse)}")
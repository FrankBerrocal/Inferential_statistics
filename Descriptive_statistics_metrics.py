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
import polars as pl
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# create df with csv the amount column.
df_financial_1D = pl.read_csv('/Users/frankberrocalazofeifa/Documents/GitHub/Inferential_statistics/Synthetic_Financial_datasets_log.csv',
                              columns=['amount'])
# create n-array using the amount column.
vector_state_1_1D = df_financial_1D['amount'].to_numpy()

# count valid elements


def nan_element_count(n_array):
    return np.sum(np.isnan(n_array))  # count of NaN, not valid

# calculate mode


def mode_calculation(n_array):
    mode_result = stats.mode(n_array, keepdims=False)
    le_mode = mode_result.mode
    frequence_du_mode = mode_result.count
    return le_mode, frequence_du_mode

# coefficient of variation


def Coefficient_variation(n_array):
    cv = 0
    cv = ((np.std(n_array)/np.mean(n_array))*100)
    return cv

# percentiles


def percentileCalculation(n_array):
    quantile25, quantile50, quantile75 = np.percentile(
        n_array, [25, 50, 75])
    return quantile25, quantile50, quantile75


# geometry
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
    mean = np.mean(n_array, axis=0)
    std = np.std(n_array, axis=0)
    return np.mean((n_array - mean) ** 3, axis=0) / std ** 3

# 4.  Pearson's Coefficient


def pearson_calculation(n_array):
    return 3 * (np.mean(n_array, axis=0) - np.median(n_array, axis=0)) / np.std(n_array, axis=0)

# Data model compression


def robust_scaling_algorithm(n_array):           # consistent name
    percentiles = np.nanpercentile(n_array, [25, 50, 75])
    q1, mediane, q3 = percentiles
    iqr = q3 - q1
    if iqr == 0:
        iqr = 1e-9
    return (n_array - mediane) / iqr

# Data model transformation


def logarithmic_transformation_algorithm(n_array):
    # use log1p to avoid logarithmic explosion when log(0)
    # forced used in finances since you can have 0 values.
    vecteur_log = np.log1p(n_array)
    return vecteur_log


# Orthogonal Feature Scaling
# Working with the complete dataset
df_financial_nD = pl.read_csv(
    '/Users/frankberrocalazofeifa/Documents/GitHub/Inferential_statistics/Synthetic_Financial_datasets_log.csv')


df = pd.DataFrame(df_financial_nD['nameOrig'])

print(df.nunique())

# 1. Initialize the Lazy Engine
optimized_request = (
    df_financial_nD.lazy()

    # 1. Extraction Dimensionnelle
    .select(['amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud'])

    # 2. Réétiquetage Métadonnées (Zéro duplication mémoire)
    .rename({
        'amount': 'Amount',
        'oldbalanceOrg': 'Old_Balance',
        'newbalanceOrig': 'New_Balance',
        'isFraud': 'Fraud_Potential'
    })
)

# 2. Execution Trigger
# The machine's query optimizer evaluates the entire plan, drops unnecessary data
# before it ever reaches RAM, and executes the math via multi-threading.
complete_vector = optimized_request.collect()

# Robust Scaling Multidimensional

# np.nanmedian(..., axis=0), np.nanpercentile(..., axis=0),
# and numpy broadcasting require a numpy array, not a Polars DataFrame.


def tensor_X_conversion(n_array):
    tensor_X = (
        n_array
        .select(['Amount', 'Old_Balance', 'New_Balance'])
        .cast(pl.Float64)
        .to_numpy()
    )
    return tensor_X


def tensor_Y_conversion(n_array):
    tensor_X = (
        n_array
        .get_column('Fraud_Potential')
        .cast(pl.Int8)
        .to_numpy()
    )
    return tensor_X


# data types conversion
x_features = tensor_X_conversion(complete_vector)
y_target = tensor_Y_conversion(complete_vector)


def robust_scaling_multiDim_algorithm(features):  # no global leak
    medians = np.nanmedian(features, axis=0)
    percentiles = np.nanpercentile(features, [25, 75], axis=0)
    iqrs = percentiles[1] - percentiles[0]
    iqrs = np.where(iqrs == 0, 1e-9, iqrs)
    return (features - medians) / iqrs


# Output report
# Measure of base information
print(f"\n\nMeasures of base information\n")
print(f"Dataset Dimensions: {vector_state_1_1D.shape}")
print(f"Total elements in set: {vector_state_1_1D.size}")
print(f"Total sum of values: {vector_state_1_1D.sum()}")
print(f"Minimum value in observations: {vector_state_1_1D.min()}")
print(f"Maximum value in observations: {vector_state_1_1D.max()}")
print(f"NaN values: {nan_element_count(vector_state_1_1D)}")


# Measures of central tendency
print(f"\n\nMeasures of central tendency\n")
print(f"Median: {np.median(vector_state_1_1D)}")
print(f"Mean: {np.mean(vector_state_1_1D)}")
print(
    f"Mode: {mode_calculation(vector_state_1_1D)[0]} with frequency {mode_calculation(vector_state_1_1D)[1]}")

# Measures of variablity
print(f"\n\nMeasures of variability\n")
print(f"Range: {np.ptp(vector_state_1_1D)}")  # difference is too low
print(f"Variance: {np.var(vector_state_1_1D)}")
# same number, no NaN values found
print(f"NaN Variance: {np.nanvar(vector_state_1_1D)}")
print(f"Standard Deviation: {np.std(vector_state_1_1D)}")
# same number, no NaN values found
print(f"NaN STD: {np.nanstd(vector_state_1_1D)}")
print(
    f"Coefficient of variation: {Coefficient_variation(vector_state_1_1D)}%")
print(f"Quantile 25%: {percentileCalculation(vector_state_1_1D)[0]}")
print(f"Quantile 50%: {percentileCalculation(vector_state_1_1D)[1]}")
print(f"Quantile 75%: {percentileCalculation(vector_state_1_1D)[2]}")

# Measures of geometry
print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(vector_state_1_1D)}")
print(f"Pearson's Coefficient: {pearson_calculation(vector_state_1_1D)}")
print(f"Skewness: {skewness_calculation(vector_state_1_1D)}")
# severe leptokurtic behavior (extreme positive kurtosis)
print(f"Kurtosis: {kurtosis_calculation(vector_state_1_1D)}")


print(f"\n\nData Model Transformation **************\n")
print(f"Logarithmic transformation\n")
vecteur_log = logarithmic_transformation_algorithm(vector_state_1_1D)

# Measures of variablity
print(f"Measures of variability\n")
print(f"Range: {np.ptp(vecteur_log)}")  # now a range is being generated
print(f"Variance: {np.var(vecteur_log)}")
print(f"Standard Deviation: {np.std(vecteur_log)}")
print(f"Conclusion:  Scale is changed.")


# Measures of geometry
print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(vecteur_log)}")
print(f"Pearson's Coefficient: {pearson_calculation(vecteur_log)}")
print(f"Skewness: {skewness_calculation(vecteur_log)}")
print(f"Kurtosis: {kurtosis_calculation(vecteur_log)}")
print(f"Conclusion:  Geometry has been changed.")
print(f"Conclusion:  Scalar and geometrical optimization")
print(f"is obtained working with the compression and transformation")
print(f"of the data, into their equivalents.\n\n")


# Data model compression
print(f"\n\nData Model Compression **************\n")
print(f"Robust Scaling\n")
vecteur_compresse = robust_scaling_algorithm(vecteur_log)

# Measures of variablity
print(f"Measures of variability\n")
print(f"Range: {np.ptp(vecteur_compresse)}")  # now a range is being generated
print(f"Variance: {np.var(vecteur_compresse)}")
print(f"Standard Deviation: {np.std(vecteur_compresse)}")
print(f"Conclusion:  Scale is changed.")


# Measures of geometry
print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(vecteur_compresse)}")
print(f"Pearson's Coefficient: {pearson_calculation(vecteur_compresse)}")
print(f"Skewness: {skewness_calculation(vecteur_compresse)}")
# severe leptokurtic behavior (extreme positive kurtosis)
print(f"Kurtosis: {kurtosis_calculation(vecteur_compresse)}")
print(f"Conclusion:  Geometry does not change.")


# Orthogonal Feature Scaling
# Logarithmic Transformation
print(f"\n\nData Model Transformation **************\n")
print(f"Logarithmic Transformation\n")
complete_vector_log = logarithmic_transformation_algorithm(x_features)

print(f"\n\nMeasures of geometry\n")
print(f"Fisher's Coefficient: {fisher_calculation(complete_vector_log)}")
print(f"Pearson's Coefficient: {pearson_calculation(complete_vector_log)}")
print(f"Skewness: {skewness_calculation(complete_vector_log)}")
print(f"Kurtosis: {kurtosis_calculation(complete_vector_log)}")

# Robust Scaling Multidimensional
print(f"\n\nData Model Compression **************\n")
print(f"Robust Scaling\n")
print(complete_vector.schema)
complete_vector_compressed = robust_scaling_multiDim_algorithm(
    complete_vector_log)

print(f"\n\nMeasures of base information\n")
print(f"N_array dimensions: {complete_vector.shape}\n")
# ← was complete_vector_log
print(
    f"Robust Scaling MultiDim compressed: {complete_vector_compressed.shape}")

print(f"Measures of variability\n")
# ← was complete_vector_log
print(f"Range: {np.ptp(complete_vector_compressed, axis=0)}")
# ← was complete_vector_log
print(f"Variance: {np.var(complete_vector_compressed, axis=0)}")
# ← was complete_vector_log
print(f"Standard Deviation: {np.std(complete_vector_compressed, axis=0)}")
print(f"Conclusion:  Scale is changed.")


# Charts
# Generate the normalized histogram canvas for the non- transformed, non-compressed Amount
# Note: density=True is mandatory for the curve's area to align with the bars
data_slice = x_features[:, 0]
plt.hist(data_slice, bins=100, density=True,
         alpha=0.6, color='g', label='Data Histogram')

# 2. Compute the non-parametric density estimation
kde = gaussian_kde(data_slice)

# 3. Create a continuous, linearly spaced axis line across the data range
x_axis = np.linspace(data_slice.min(), data_slice.max(), 1000)

# 4. Project the curve over the histogram
plt.plot(x_axis, kde(x_axis), color='darkgreen',
         linewidth=2, label='KDE Curve')

plt.title('Amount Profile')
plt.xlabel('Value Space')
plt.ylabel('Density')
plt.legend()
plt.show()


# Generate the normalized histogram canvas for the transformed and compressed Amount
data_slice2 = complete_vector_compressed[:, 0]
plt.hist(data_slice2, bins=5, density=True, alpha=0.6,
         color='b', label='Data Histogram')

plt.title('Distribution of Amount after Transformation and Compression')
plt.xlabel('Value Space')
plt.ylabel('Density')
plt.legend()
plt.show()


print("\n=== Initializing Dimensional Split ===")
# 1. The Matrix Schism (80% Training / 20% Validation)
# We map the scaled 2D matrix (X) to the pure 1D target vector (Y)
X_train, X_test, y_train, y_test = train_test_split(
    complete_vector_compressed,
    y_target,
    test_size=0.20,
    random_state=42,
    stratify=y_target  # CRITICAL: Ensures the rare fraud cases are evenly distributed
)


print(f"Training Tensor Shape: {X_train.shape}")
print(f"Validation Tensor Shape: {X_test.shape}")

# configure to test outcomes.  I cannot run the optimization due to RAM is deleted and future state cannot be calculated.
# Use Jupyter Notebooks
# neutral, fraud is 100 times painful than false alarms, 1-5 is a low penalty.
scale_pos = 1

# 2. Engine Parameterization
# XGBoost is highly optimized for imbalanced, spiked, tabular datasets
xgb_engine = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',       # Hardware acceleration for large continuous matrices
    max_depth=6,              # Limits the geometric depth to prevent overfitting
    learning_rate=0.1,        # The step size for gradient descent
    # Mathematical penalty multiplier for missing a fraud case
    scale_pos_weight=scale_pos,
    random_state=42
)

# 3. Model Training (The Algorithmic Mapping)
print("\n=== Executing Gradient Boosting Optimization... ===")
xgb_engine.fit(X_train, y_train)

# 4. Target Prediction Space
print("=== Generating Vector Predictions ===")
y_predictions = xgb_engine.predict(X_test)

# 5. Geometrical Efficacy Report
print("\n=== Supervised Learning Output Metrics ===")
print(confusion_matrix(y_test, y_predictions))
print("\n")
print(classification_report(y_test, y_predictions))

Interpretable Machine Learning for RO/NF Membrane Desalination

This repository contains the curated dataset and Python scripts for the paper:
Unveiling the Dominance of Operating Conditions over Membrane Chemistry in RO/NF Desalination: A SHAP-Based Interpretability Analysis

Dataset
The cleaned_OMD_data.csv file contains the final curated dataset used for model training. It was extracted and rigorously cleaned from the Open Membrane Database (OMD). The final dataset comprises 540 valid reverse osmosis (RO) and nanofiltration (NF) membrane records, with comprehensive feature engineering (e.g., median imputation for missing CP values, multi-hot encoding for membrane chemistry).

Requirements
Ensure you have Python 3.8+ installed. Install the required packages using:
pip install pandas numpy matplotlib seaborn xgboost scikit-learn shap

How to Run
Run the analysis script directly from the command line. It will train the XGBoost models and generate all SHAP, PDP, and SI figures cited in the paper:
python omd_xgb_shap_analysis.py

Outputs
The script will generate the following high-resolution figures (300 DPI):

shap_summary_Robs.png: Global SHAP summary plot for salt rejection.

shap_dependence_hydraulicP_bar.png: SHAP dependence for pressure.

shap_dependence_NaCl_M.png: SHAP dependence for salinity.

shap_dependence_CP.png: SHAP dependence for concentration polarization.

pdp_2D_Pressure_NaCl.png: 2D Partial Dependence Plot (Pressure x Salinity).

SI_Data_Distribution.png: Sampling distribution histograms for Supporting Information.

Citation
If you use this data or code in your work, please cite the corresponding paper (under review).


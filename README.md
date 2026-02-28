# SHAP Values and Other Indicators of Feature Predictive Power in Binary Classification 



## 📌 Project Overview

This repository contains the implementation, analysis, and experiments from the thesis *SHAP Values and Other Indicators of Feature Predictive Power in Binary Classification*. The primary objective of this project is to conduct a comparative analysis of feature predictive power assessment methods, evaluating the reliability of SHAP (SHapley Additive explanations) values against Permutation Feature Importance (PFI) and standard statistical measures like Pearson Correlation and Normalized Mutual Information (NMI).

The study explores these methods in the context of Logistic Regression and XGBoost models, specifically highlighting how they behave under extreme multicollinearity.



## 🔬 Methodology

### Interpretability Methods Analyzed

* **Pearson Correlation Coefficient**: Serves as the baseline for assessing linear associations between variables.


* **Normalized Mutual Information (NMI)**: An information-theoretic approach designed to detect the magnitude of shared relationships between two random variables, capable of capturing non-linear and non-monotonic dependencies.


* **Permutation Feature Importance (PFI)**: A model-agnostic technique that evaluates importance by measuring the degradation in model performance (measured via ROC AUC) when a feature's values are randomly shuffled.


* **SHAP (SHapley Additive explanations)**: A unified framework rooted in cooperative game theory that distributes a model's prediction payout among its features, satisfying desirable axiomatic properties (Efficiency, Symmetry, Dummy Player, Additivity).



### Datasets

1. **Breast Cancer Wisconsin (Diagnostic) Dataset**: A strictly numerical dataset of 569 samples and 30 features derived from digitized images of cell nuclei. It is characterized by severe multicollinearity (e.g., radius, perimeter, and area are mathematically related).


2. **Adult Census Income Dataset**: Contains 32,561 records with a mix of continuous and categorical features, predicting whether an individual earns more than $50,000 annually.



### Models

* **Logistic Regression**: Implemented as a linear baseline model.


* **XGBoost (Extreme Gradient Boosting)**: Implemented as a complex, non-linear _"black-box"_ tree ensemble.


## 📊 Key Findings and Experiments

### 1. The Impact of Multicollinearity

In the Breast Cancer Wisconsin dataset, strong near-perfect linear relationships were observed, such as between mean radius and mean perimeter ($r=0.9979$).

<div align="center" style="margin-bottom: 100px;">
  <img src="images/corr_plots.png" alt="Correlation Plots" width="100%">
  <p><em>Pairwise relationships between selected features from the Breast Cancer Wisconsin dataset. Below-diagonal panels show scatter plots colored by diagnosis (red=malignant, blue=benign). We can observe almost perfect collinearity between mean radius and mean perimeter, and exponential-like relationships among mean area vs. mean radius and mean area vs. mean perimeter. Diagonal panels display kernel density estimates divided by class. For each feature pair, Pearson Correlation Coefficients (Corr) and Normalized Mutual Information (NMI) are reported.</em></p>
</div>

### 2. Discrepancies Between SHAP and PFI

Experimental results reveal significant inconsistencies between global rankings derived from PFI and SHAP.

* **Substitution Effect**: In highly correlated environments, PFI scores were found to be uniformly low because models could rely on correlated substitute features when one was permuted.


* **Model Architecture Matters**: Logistic Regression distributed its feature importance across a broader set of correlated features (showing a funnel-shaped SHAP decay), while XGBoost performed internal feature selection, assigning near-zero importance to non-informative features.


<div align="center" style="margin-bottom: 100px;">
  <img src="images/image_517da1.png" alt="image_517da1" width="100%">
  <p><em>Permutation Feature Importance (PFI) for Breast Cancer Dataset. The x-axis represents the drop in ROC AUC score when feature values are permuted.</em></p>
</div>



<div align="center" style="margin-bottom: 100px;">
  <img src="images/adult_pfi.png" alt="adult_pfi" width="100%">
  <p><em>Permutation Feature Importance (PFI) for Adult Census Income Dataset. The x-axis represents the drop in ROC AUC score when feature values are permuted.</em></p>
</div>




<div align="center" style="margin-bottom: 100px;">
  <img src="images/image_516b1e.png" alt="image_516b1e" width="100%">
  <p><em>SHAP summary plots for Breast Cancer Dataset. Points represent individual instances; color indicates feature value (red=high, blue=low); x-axis displays the SHAP values. High values of worst texture (red) result in negative SHAP values, increasing the predicted risk of malignancy.</em></p>
</div>




<div align="center" style="margin-bottom: 100px;">
  <img src="images/image_51677d1.png" alt="image_51677d1" width="100%">
  <p><em>SHAP summary plots for Adult Census Income Dataset. Each point represents a single observation.</em></p>
</div>


### 3. Iterative Feature Addition

Iterative selection experiments (Forward and Backward Selection) demonstrated that SHAP does not necessarily prioritize features with the highest standalone predictive power in redundant datasets. When features were added sequentially based on rankings, PFI and Correlation sometimes recovered model performance faster than SHAP in the initial stages.

<div align="center" style="margin-bottom: 100px;">
  <img src="images/breast_best.png" alt="breast_best" width="100%">
  <p><em>Iterative Feature Addition Curves (Forward Ranking) for Breast Cancer Wisconsin Dataset (Logistic Regression).</em></p>
</div>



<div align="center" style="margin-bottom: 100px;">
  <img src="images/adult_best.png" alt="adult_best" width="100%">
  <p><em>Iterative Feature Addition Curves (Forward Ranking) for Adult Census Income Dataset (XGBoost).</em></p>
</div>

### 4. Mitigating Multicollinearity via Hierarchical Clustering

To stabilize importance rankings and address the substitution effect, a dimensionality reduction strategy using Hierarchical Clustering (Ward's minimum variance method) was applied to the feature space.

* **Dimensionality Reduction**: The feature space was reduced from 30 variables to 11 representative features.


* **Performance Maintenance**: The reduced Logistic Regression model achieved near-identical predictive power (ROC AUC of 0.9967 vs 0.9974) compared to the full model.


* **Convergence of Interpretability**: The removal of redundancy unmasked the true predictive power of features, causing PFI scores to surge and bringing SHAP and PFI rankings into alignment.
  

<div align="center" style="margin-bottom: 100px;">
  <table>
    <thead>
      <tr>
        <th align="left">Metric</th>
        <th align="left">Full Model (30 Features)</th>
        <th align="left">Reduced Model (11 Features, $t=0.5$)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td align="left"><strong>Precision</strong></td>
        <td align="left">0.98</td>
        <td align="left">0.98</td>
      </tr>
      <tr>
        <td align="left"><strong>Recall</strong></td>
        <td align="left">0.95</td>
        <td align="left">0.95</td>
      </tr>
      <tr>
        <td align="left"><strong>F1-Score</strong></td>
        <td align="left">0.96</td>
        <td align="left">0.96</td>
      </tr>
      <tr>
        <td align="left"><strong>Accuracy</strong></td>
        <td align="left">0.97</td>
        <td align="left">0.97</td>
      </tr>
      <tr>
        <td align="left"><strong>ROC AUC</strong></td>
        <td align="left">0.9974</td>
        <td align="left">0.9967</td>
      </tr>
    </tbody>
  </table>
  <p><em> Performance Comparison: Full Feature Set vs. Reduced Feature Set at $t=0.5$ (class: Malignant)</em></p>
</div>

<div align="center" style="margin-bottom: 100px;">
  <img src="images/dendogram.png" alt="dendogram" width="100%">
  <p><em>Hierarchical Clustering Dendrogram (Ward Linkage). The y-axis represents the Ward linkage distance (increase in within-cluster variance). The black dashed line at threshold $t=0.5$ cuts the tree into 11 distinct clusters by grouping highly redundant features.</em></p>
</div>



<div align="center" style="margin-bottom: 100px;">
  <img src="images/pfi_after.png" alt="pfi_after" width="100%">
  <p><em>Permutation Feature Importance (PFI) for the Reduced Model. We can observe more pronounced importance scores compared to the original model. All retained predictors now have positive PFI scores.</em></p>
</div>



<div align="center" style="margin-bottom: 100px;">
  <img src="images/shap_after.png" alt="shap_after" width="100%">
  <p><em>SHAP Summary Plot for the Reduced Model (11 features). Compared to the full model, the SHAP values here exhibit substantially higher magnitude (maximum $\approx 2.23$ vs $0.96$) and a wider spread, facilitating improved interpretability.</em></p>
</div>



## 💡 Conclusions

* **Linear vs. Non-linear Dependencies**: Pearson Correlation is an effective baseline for linear dependencies, but NMI is required for categorical or non-linear relationships (like those in the Adult dataset).


* **Ranking Inconsistencies**: Multicollinearity heavily obscures feature importance. PFI tends to underestimate importance due to the substitution effect, while SHAP distributes credit among correlated variables.


* **Clustering Enhances Reliability**: Grouping highly correlated features prior to model training significantly enhances the stability and clinical reliability of feature importance assessments

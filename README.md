# Random Forest Implementation from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of Random Forests from scratch, including theoretical foundations, custom implementations, and extensive experiments on real-world datasets.

## 📋 Project Overview

This project addresses **Question 2** from the Machine Learning Lab Mid-term examination, which requires:

- **Part A**: Theoretical explanation of Random Forest key ingredients
- **Part B**: Implementation and experiments on two datasets
- **Part C**: Comprehensive report with analysis and insights

### Key Features

✅ **Custom Implementation**: Decision Tree and Random Forest built from scratch with full mathematical formulations  
✅ **Theoretical Foundation**: Detailed explanations based on Leo Breiman's 2001 paper  
✅ **Dual Experiments**: Heart Disease UCI (tabular) and Intel Image Classification (image)  
✅ **Comprehensive Analysis**: Accuracy plots, confusion matrices, feature importance  
✅ **Production-Ready Code**: Clean, documented, and reproducible  

## 🎯 Objectives

1. Understand and explain the two key ingredients of Random Forests:
   - Strength of individual trees
   - Low correlation between trees

2. Conduct experiments to explore:
   - Effect of varying number of trees (n_estimators)
   - Comparison between Decision Tree and Random Forest
   - Impact on accuracy, stability, and training time

3. Provide insights on:
   - How randomness affects model performance
   - Optimal ensemble size
   - Practical trade-offs

## 📁 Project Structure

```
Build-Random-Forests-From-scratch/
│
├── README.md                              # This file
├── REPORT.md                              # Comprehensive report (Part C)
├── THEORY.md                              # Theoretical foundation (Part A)
├── BREIMAN_SUMMARY.md                     # Summary of Breiman's 2001 paper
│
├── requirements.txt                       # Python dependencies
├── config.py                              # Configuration and parameters
│
├── decision_tree_scratch.py               # Decision Tree from scratch
├── random_forest_scratch.py               # Random Forest from scratch
│
├── utils.py                               # Data loading and preprocessing
├── visualization.py                       # Plotting utilities
│
├── experiments_heart_disease.py           # Experiments on Heart Disease dataset
├── experiments_image_classification.py    # Experiments on Image dataset
├── test_custom_implementation.py          # Testing script
│
├── JSON_RESULTS_STRUCTURE.md              # JSON results documentation
│
├── data/                                  # Datasets (downloaded automatically)
│   ├── heart_disease/
│   └── intel_images/
│
└── outputs/                               # Results and plots
    ├── plots/                             # Generated visualizations
    └── results/                           # Experiment results
        ├── *.pkl                          # Pickle format (Python)
        └── *.json                         # JSON format (human-readable)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/codewithdark-git/Build-Random-Forests-From-scratch.git
cd Build-Random-Forests-From-scratch
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running Experiments

#### Heart Disease UCI Dataset

  - Strength of individual trees
  - Low correlation between trees
  - Mathematical formulations
  - Intuitive explanations

- **[BREIMAN_SUMMARY.md](BREIMAN_SUMMARY.md)**: Summary of Breiman's 2001 paper
  - Main contributions
  - Theoretical guarantees
  - Key insights

- **[REPORT.md](REPORT.md)**: Comprehensive report (Part C)
  - Experimental setup
  - Results and analysis
  - Discussion and insights
  - Conclusions

### Code Documentation

All Python files include:
- Docstrings for classes and functions
- Mathematical formulations in comments
- Type hints for clarity
- Usage examples

## 🔬 Implementation Details

### Decision Tree

**Features**:
- Gini impurity and entropy splitting criteria
- Configurable max depth, min samples split/leaf
- Feature randomization support
- Prediction and probability estimation

**Mathematical Foundations**:
- Gini: $Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$
- Entropy: $H(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$
- Information Gain: $IG = H(D) - \sum_v \frac{|D_v|}{|D|} H(D_v)$

### Random Forest

**Features**:
- Bootstrap sampling (bagging)
- Random feature selection at each split
- Parallel tree building (optional)
- Out-of-bag error estimation
- Feature importance calculation

**Mathematical Foundations**:
- Generalization error: $PE^* \leq \bar{\rho} \frac{(1-s^2)}{s^2}$
- Ensemble prediction: $\hat{y} = \text{mode}(\{h_1(x), ..., h_T(x)\})$
- OOB error: Unbiased estimate using ~36.8% out-of-bag samples

## 🎓 Key Insights

### Effect of Number of Trees

1. **Accuracy**: Improves rapidly initially, then plateaus
2. **Stability**: Variance decreases as $1/T$
3. **Optimal Range**: 100-300 trees for most applications
4. **No Overfitting**: Can add trees without hurting generalization

### Random Forest vs Decision Tree

1. **Accuracy**: RF consistently outperforms (10-15% improvement)
2. **Generalization**: RF reduces overfitting significantly
3. **Robustness**: RF more resistant to noise and outliers
4. **Trade-off**: Slightly longer training time, less interpretable

### Randomness and Ensemble Size

1. **Bootstrap Sampling**: Creates diversity in training data
2. **Feature Randomization**: Decorrelates trees effectively
3. **Optimal $m$**: $\sqrt{p}$ for classification balances strength and correlation
4. **Ensemble Size**: Logarithmic accuracy improvement with tree count

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Random seed for reproducibility
RANDOM_STATE = 42

# Experiment parameters
N_ESTIMATORS_RANGE = [1, 10, 50, 100, 300]
TEST_SIZE = 0.2

# Random Forest parameters
RF_MAX_FEATURES = 'sqrt'
RF_BOOTSTRAP = True
RF_OOB_SCORE = True

# Image processing
IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 1000
```

## 📊 Visualizations

All plots are automatically generated and saved to `outputs/plots/`:

1. **Accuracy vs n_estimators**: Shows convergence behavior
2. **Training time comparison**: Linear scaling verification
3. **Tree vs Forest comparison**: Bar charts of metrics
4. **Confusion matrices**: Error analysis
5. **Feature importance**: Top contributing features
6. **Summary plots**: Comprehensive 4-panel overview

## 🧩 Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and metrics
- **matplotlib**: Plotting
- **seaborn**: Statistical visualizations
- **Pillow**: Image processing
- **joblib**: Parallel processing

See `requirements.txt` for specific versions.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@codewithdark-git](https://github.com/codewithdark-git)
- Email: codewithdark@gmail.com

## 🙏 Acknowledgments

- **Leo Breiman** for the Random Forests algorithm
- **UCI Machine Learning Repository** for the Heart Disease dataset
- **Kaggle** for the Intel Image Classification dataset
- **Scikit-learn** for reference implementations

## 📖 References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
3. Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice*.

---

**Note**: This project is created for educational purposes as part of a Machine Learning course assignment (Question 2 - Random Forests).

**Last Updated**: December 2025

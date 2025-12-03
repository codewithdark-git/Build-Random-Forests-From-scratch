"""
Random Forest Implementation from Scratch

This module implements a Random Forest classifier with detailed mathematical formulations.
Includes bootstrap sampling, feature randomization, and out-of-bag error estimation.
Optimized for performance with vectorized prediction and warm start support.

Mathematical Foundations:
-----------------------

1. Bootstrap Sampling:
   For each tree, create a bootstrap sample by sampling n samples with replacement
   from the training set. Approximately 63.2% of samples are unique in each bootstrap.
   
   P(sample not selected) = (1 - 1/n)^n → 1/e ≈ 0.368 as n → ∞
   P(sample selected) ≈ 0.632

2. Feature Randomization:
   At each split, randomly select m features from p total features:
   - Classification: m = √p (default)
   - Regression: m = p/3 (default)

3. Ensemble Prediction:
   For classification, use majority voting:
   ŷ = mode({h₁(x), h₂(x), ..., h_T(x)})
   
   For probability estimation:
   P(y = k | x) = (1/T) Σ I(h_t(x) = k)

4. Out-of-Bag (OOB) Error:
   For each sample, predict using only trees where it was not in bootstrap sample:
   OOB_Error = (1/n) Σ I(y_i ≠ ŷ_i^OOB)

5. Generalization Error Bound (Breiman 2001):
   PE* ≤ ρ̄ * (1-s²)/s²
   where:
   - ρ̄ = average correlation between trees
   - s = strength of individual trees
"""

import numpy as np
from collections import Counter
from typing import Optional, List, Tuple
from decision_tree_scratch import DecisionTree
from joblib import Parallel, delayed
from scipy import stats


class RandomForest:
    """
    Random Forest Classifier implemented from scratch.
    
    Implements the Random Forest algorithm as described by Leo Breiman (2001).
    Uses bootstrap aggregating (bagging) and random feature selection to create
    an ensemble of decorrelated decision trees.
    
    Optimizations:
    - Vectorized prediction
    - Warm start support
    - Memory optimized bootstrap storage
    - Min impurity decrease support
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest.
    
    max_depth : int, default=None
        Maximum depth of each tree. If None, nodes are expanded until pure.
    
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    
    max_features : int, float, str, or None, default='sqrt'
        Number of features to consider for best split:
        - If 'sqrt': sqrt(n_features)
        - If 'log2': log2(n_features)
        - If int: max_features features
        - If float: int(max_features * n_features)
        - If None: all features
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate generalization error.
    
    n_jobs : int, default=None
        Number of jobs to run in parallel. -1 means using all processors.
    
    random_state : int, default=None
        Random seed for reproducibility.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    
    criterion : str, default='gini'
        Split quality measure. Supported: 'gini', 'entropy'
    
    Attributes:
    -----------
    trees_ : list of DecisionTree
        The collection of fitted sub-estimators.
    
    n_features_ : int
        Number of features seen during fit.
    
    n_classes_ : int
        Number of classes.
    
    classes_ : array of shape (n_classes,)
        The class labels.
    
    oob_score_ : float
        Score of the training dataset obtained using out-of-bag estimate.
        Only available if oob_score=True.
    
    oob_decision_function_ : array of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate.
        Only available if oob_score=True.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        criterion: str = 'gini'
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start
        self.criterion = criterion
        
        self.trees_ = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self._bootstrap_indices = []
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Build a forest of trees from training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : RandomForest
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Handle classes
        if self.classes_ is None or not self.warm_start:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        
        # Clear existing trees if not warm_start
        if not self.warm_start:
            self.trees_ = []
            self._bootstrap_indices = []
        
        # Determine how many trees to add
        n_more_estimators = self.n_estimators - len(self.trees_)
        
        if n_more_estimators <= 0:
            print(f"Warm-start: no new trees to build (current: {len(self.trees_)}, target: {self.n_estimators})")
            return self
        
        # Generate random seeds for new trees
        seeds = self.rng.randint(0, 10000, size=n_more_estimators)
        
        # Build trees in parallel or sequentially
        if self.n_jobs is not None and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._build_tree)(X, y, seed, i)
                for i, seed in enumerate(seeds)
            )
            new_trees = [tree for tree, _ in results]
            new_indices = [indices for _, indices in results]
        else:
            new_trees = []
            new_indices = []
            for i, seed in enumerate(seeds):
                tree, indices = self._build_tree(X, y, seed, i)
                new_trees.append(tree)
                new_indices.append(indices)
        
        self.trees_.extend(new_trees)
        if self.oob_score:
            self._bootstrap_indices.extend(new_indices)
        
        # Calculate OOB score if requested
        if self.oob_score:
            self._calculate_oob_score(X, y)
        
        return self
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int,
        tree_idx: int
    ) -> Tuple[DecisionTree, Optional[np.ndarray]]:
        """
        Build a single decision tree.
        
        Parameters:
        -----------
        X : array of shape (n_samples, n_features)
            Training data
        y : array of shape (n_samples,)
            Target values
        seed : int
            Random seed for this tree
        tree_idx : int
            Index of tree being built
        
        Returns:
        --------
        tree : DecisionTree
            Fitted decision tree
        bootstrap_indices : array or None
            Indices of samples used in bootstrap (None if oob_score=False)
        """
        n_samples = X.shape[0]
        
        # Bootstrap sampling
        if self.bootstrap:
            rng = np.random.RandomState(seed)
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[bootstrap_indices]
            y_sample = y[bootstrap_indices]
        else:
            bootstrap_indices = np.arange(n_samples)
            X_sample = X
            y_sample = y
        
        # Create and fit tree
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=seed
        )
        tree.fit(X_sample, y_sample)
        
        # Only return indices if needed for OOB score to save memory
        return tree, bootstrap_indices if self.oob_score else None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Uses majority voting across all trees in the forest.
        Optimized with vectorized operations.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        
        # Get predictions from all trees: (n_estimators, n_samples)
        all_predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        # Transpose to (n_samples, n_estimators)
        all_predictions = all_predictions.T
        
        # Majority voting using mode
        # mode returns (mode_val, count)
        mode_result = stats.mode(all_predictions, axis=1, keepdims=False)
        
        # Handle different scipy versions
        if isinstance(mode_result, tuple):
             y_pred = mode_result[0]
        else:
             y_pred = mode_result.mode
             
        return y_pred.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Probability is calculated as the proportion of trees voting for each class.
        Optimized with vectorized operations.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Get predictions from all trees: (n_estimators, n_samples)
        all_predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        # Calculate probabilities
        probas = np.zeros((n_samples, self.n_classes_))
        
        # Vectorized counting
        # Iterate over classes and count how many trees predicted that class
        for i, class_label in enumerate(self.classes_):
            # (n_estimators, n_samples) == scalar -> boolean array
            # sum over axis 0 (estimators) -> (n_samples,)
            votes = np.sum(all_predictions == class_label, axis=0)
            probas[:, i] = votes / len(self.trees_)
        
        return probas
    
    def _calculate_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate out-of-bag score.
        
        For each sample, make predictions using only trees where the sample
        was not included in the bootstrap sample (out-of-bag samples).
        
        Parameters:
        -----------
        X : array of shape (n_samples, n_features)
            Training data
        y : array of shape (n_samples,)
            Target values
        """
        n_samples = X.shape[0]
        
        # Initialize OOB decision function
        oob_decision = np.zeros((n_samples, self.n_classes_))
        n_oob_predictions = np.zeros(n_samples)
        
        # For each tree, predict on its OOB samples
        for tree_idx, (tree, bootstrap_indices) in enumerate(
            zip(self.trees_, self._bootstrap_indices)
        ):
            if bootstrap_indices is None:
                continue
                
            # Find OOB samples (not in bootstrap)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[bootstrap_indices] = False
            oob_indices = np.where(oob_mask)[0]
            
            if len(oob_indices) == 0:
                continue
            
            # Predict on OOB samples
            oob_predictions = tree.predict(X[oob_indices])
            
            # Update decision function
            # Map predictions to class indices
            # This part is still a bit slow due to loop, but OOB is optional
            for idx, pred in zip(oob_indices, oob_predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                oob_decision[idx, class_idx] += 1
                n_oob_predictions[idx] += 1
        
        # Normalize to get probabilities
        valid_oob = n_oob_predictions > 0
        oob_decision[valid_oob] /= n_oob_predictions[valid_oob, np.newaxis]
        
        # Calculate OOB score (accuracy)
        if np.any(valid_oob):
            oob_predictions = self.classes_[np.argmax(oob_decision, axis=1)]
            self.oob_score_ = np.mean(oob_predictions[valid_oob] == y[valid_oob])
        else:
            self.oob_score_ = 0.0
            
        self.oob_decision_function_ = oob_decision
    
    def feature_importances(self) -> np.ndarray:
        """
        Calculate feature importances based on mean decrease in impurity.
        
        Feature importance is calculated as the total reduction in node impurity
        (weighted by probability of reaching that node) averaged over all trees.
        
        Returns:
        --------
        importances : array of shape (n_features,)
            Feature importances (normalized to sum to 1)
        """
        importances = np.zeros(self.n_features_)
        
        for tree in self.trees_:
            tree_importances = self._get_tree_feature_importances(tree)
            importances += tree_importances
        
        # Average over trees
        importances /= len(self.trees_)
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances
    
    def _get_tree_feature_importances(self, tree: DecisionTree) -> np.ndarray:
        """
        Calculate feature importances for a single tree.
        
        Parameters:
        -----------
        tree : DecisionTree
            Decision tree
        
        Returns:
        --------
        importances : array of shape (n_features,)
            Feature importances for this tree
        """
        importances = np.zeros(self.n_features_)
        self._accumulate_importances(tree.root, importances)
        return importances
    
    def _accumulate_importances(self, node, importances: np.ndarray) -> int:
        """
        Recursively accumulate feature importances from tree nodes.
        
        Parameters:
        -----------
        node : DecisionTreeNode
            Current node
        importances : array of shape (n_features,)
            Array to accumulate importances
        
        Returns:
        --------
        n_samples : int
            Number of samples at this node
        """
        if node is None or node.is_leaf():
            return node.samples if node is not None else 0
        
        # Get samples in left and right children
        n_left = self._accumulate_importances(node.left, importances)
        n_right = self._accumulate_importances(node.right, importances)
        n_node = node.samples
        
        # Calculate importance contribution
        # Importance = (n_node / n_total) * (impurity - weighted_child_impurity)
        if n_node > 0:
            weighted_child_impurity = (n_left / n_node) * node.left.gini + \
                                     (n_right / n_node) * node.right.gini
            importance = n_node * (node.gini - weighted_child_impurity)
            importances[node.feature_idx] += importance
        
        return n_node
    
    def get_params(self) -> dict:
        """
        Get parameters of the Random Forest.
        
        Returns:
        --------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'warm_start': self.warm_start,
            'criterion': self.criterion
        }
    
    def __str__(self) -> str:
        """String representation of Random Forest."""
        return f"RandomForest(n_estimators={self.n_estimators}, " \
               f"max_features={self.max_features}, " \
               f"max_depth={self.max_depth})"
    
    def __repr__(self) -> str:
        """Detailed representation of Random Forest."""
        params = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"RandomForest({params})"

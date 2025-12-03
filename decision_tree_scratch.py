"""
Decision Tree Implementation from Scratch

This module implements a Decision Tree classifier with detailed mathematical formulations.
Includes Gini impurity, entropy, and information gain calculations.
Optimized with vectorized operations and indices-based recursion.

Mathematical Foundations:
-----------------------

1. Gini Impurity:
   Gini(D) = 1 - Σ(p_k²) for k in classes
   where p_k is the proportion of class k in dataset D

2. Entropy:
   H(D) = -Σ(p_k * log₂(p_k)) for k in classes
   
3. Information Gain:
   IG(D, feature) = H(D) - Σ((|D_v| / |D|) * H(D_v))
   where D_v is the subset of D where feature = v

4. Best Split:
   For continuous features, find threshold t that maximizes:
   IG(D, feature, t) = H(D) - (|D_left|/|D|)*H(D_left) - (|D_right|/|D|)*H(D_right)
"""

import numpy as np
from collections import Counter
from typing import Optional, Tuple, Any, List, Dict


class DecisionTreeNode:
    """
    Node in a decision tree.
    
    Attributes:
        feature_idx: Index of feature to split on (None for leaf nodes)
        threshold: Threshold value for splitting (None for leaf nodes)
        left: Left child node
        right: Right child node
        value: Class label for leaf nodes (None for internal nodes)
        gini: Gini impurity at this node
        samples: Number of samples at this node
        class_distribution: Distribution of classes at this node
    """
    
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['DecisionTreeNode'] = None,
        right: Optional['DecisionTreeNode'] = None,
        value: Optional[Any] = None,
        gini: float = 0.0,
        samples: int = 0,
        class_distribution: Optional[dict] = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
        self.samples = samples
        self.class_distribution = class_distribution or {}
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier implemented from scratch.
    
    This implementation uses CART (Classification and Regression Trees) algorithm
    with Gini impurity as the default splitting criterion.
    
    Optimizations:
    - Vectorized split finding
    - Indices-based recursion (no data copying)
    - Min impurity decrease pruning
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    
    criterion : str, default='gini'
        Function to measure split quality. Supported: 'gini', 'entropy'
    
    max_features : int, float, str, or None, default=None
        Number of features to consider when looking for best split:
        - If int: consider max_features features
        - If float: consider int(max_features * n_features) features
        - If 'sqrt': consider sqrt(n_features) features
        - If 'log2': consider log2(n_features) features
        - If None: consider all features
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        max_features: Optional[Any] = None,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Build decision tree from training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : DecisionTree
            Fitted estimator
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Build the tree using indices to avoid copying data
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        self.root = self._build_tree(X, y, indices, depth=0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
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
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
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
        probas = []
        
        for x in X:
            node = self._traverse_tree_to_leaf(x, self.root)
            proba = np.zeros(self.n_classes_)
            for class_idx, class_label in enumerate(self.classes_):
                proba[class_idx] = node.class_distribution.get(class_label, 0) / node.samples
            probas.append(proba)
        
        return np.array(probas)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> DecisionTreeNode:
        """
        Recursively build decision tree using indices.
        
        Parameters:
        -----------
        X : array of shape (n_samples, n_features)
            Full training data
        y : array of shape (n_samples,)
            Full target values
        indices : array
            Indices of samples at current node
        depth : int
            Current depth in tree
        
        Returns:
        --------
        node : DecisionTreeNode
            Root of subtree
        """
        n_samples = len(indices)
        node_y = y[indices]
        n_labels = len(np.unique(node_y))
        
        # Calculate class distribution and impurity
        class_counts = Counter(node_y)
        class_distribution = dict(class_counts)
        
        if self.criterion == 'gini':
            impurity = self._calculate_gini(node_y)
        else:
            impurity = self._calculate_entropy(node_y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            return DecisionTreeNode(
                value=self._most_common_label(node_y),
                gini=impurity,
                samples=n_samples,
                class_distribution=class_distribution
            )
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split_vectorized(X, y, indices, impurity)
        
        # Check min_impurity_decrease
        if best_feature is None or best_gain < self.min_impurity_decrease:
            return DecisionTreeNode(
                value=self._most_common_label(node_y),
                gini=impurity,
                samples=n_samples,
                class_distribution=class_distribution
            )
        
        # Split indices
        # Note: We re-compute the mask here. We could return it from _find_best_split but that might be complex with vectorization.
        # Since we know the best feature and threshold, this is fast O(N).
        feature_values = X[indices, best_feature]
        left_mask = feature_values <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        # Check minimum samples per leaf
        if len(left_indices) < self.min_samples_leaf or \
           len(right_indices) < self.min_samples_leaf:
            return DecisionTreeNode(
                value=self._most_common_label(node_y),
                gini=impurity,
                samples=n_samples,
                class_distribution=class_distribution
            )
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X, y, left_indices, depth + 1)
        right_child = self._build_tree(X, y, right_indices, depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gini=impurity,
            samples=n_samples,
            class_distribution=class_distribution
        )
    
    def _find_best_split_vectorized(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        indices: np.ndarray, 
        current_impurity: float
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold to split on using vectorized operations.
        
        Parameters:
        -----------
        X : array
            Full training data
        y : array
            Full target values
        indices : array
            Indices of samples at current node
        current_impurity : float
            Impurity of current node
            
        Returns:
        --------
        best_feature : int
        best_threshold : float
        best_gain : float
        """
        n_samples = len(indices)
        n_features = X.shape[1]
        
        if n_samples <= 1:
            return None, None, 0.0
        
        # Get feature indices to consider
        feature_indices = self._get_feature_indices(n_features)
        
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        
        # Pre-compute one-hot encoding of y for the current node
        # Map classes to 0..K-1 integers for array indexing
        node_y = y[indices]
        unique_classes, y_encoded = np.unique(node_y, return_inverse=True)
        n_classes = len(unique_classes)
        
        # One-hot: (n_samples, n_classes)
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_encoded] = 1
        
        # Total counts per class
        total_class_counts = np.sum(y_one_hot, axis=0)
        
        for feature_idx in feature_indices:
            # Get feature values for current node
            feature_values = X[indices, feature_idx]
            
            # Sort samples by feature value
            sort_idx = np.argsort(feature_values)
            sorted_y_one_hot = y_one_hot[sort_idx]
            sorted_feature_values = feature_values[sort_idx]
            
            # Calculate cumulative class counts
            # left_counts[i] = sum of classes for samples 0..i
            left_counts = np.cumsum(sorted_y_one_hot, axis=0)
            
            # right_counts = total - left
            right_counts = total_class_counts - left_counts
            
            # Number of samples in left and right
            n_left = np.arange(1, n_samples + 1)
            n_right = n_samples - n_left
            
            # Avoid division by zero and enforce min_samples_leaf
            # Valid split points are where n_left >= min_leaf AND n_right >= min_leaf
            valid_mask = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            
            # Also, we should only split between different feature values
            # If sorted_feature_values[i] == sorted_feature_values[i+1], we can't split there
            # Shifted difference
            if n_samples > 1:
                diff_values = sorted_feature_values[1:] != sorted_feature_values[:-1]
                # Pad with False at the end to match length
                diff_values = np.append(diff_values, False)
                valid_mask = valid_mask & diff_values
            
            if not np.any(valid_mask):
                continue
            
            # Calculate impurity for all splits
            # Gini = 1 - sum(p^2)
            
            # Left impurity
            # p_left = left_counts / n_left[:, None]
            # gini_left = 1 - sum(p_left^2)
            gini_left = 1.0 - np.sum((left_counts[valid_mask] / n_left[valid_mask][:, None]) ** 2, axis=1)
            
            # Right impurity
            gini_right = 1.0 - np.sum((right_counts[valid_mask] / n_right[valid_mask][:, None]) ** 2, axis=1)
            
            # Weighted impurity
            weighted_impurity = (n_left[valid_mask] / n_samples) * gini_left + \
                                (n_right[valid_mask] / n_samples) * gini_right
            
            # Information gain
            gains = current_impurity - weighted_impurity
            
            if len(gains) > 0:
                max_gain_idx = np.argmax(gains)
                max_gain = gains[max_gain_idx]
                
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_feature = feature_idx
                    
                    # Find the threshold corresponding to the best split
                    # The split index corresponds to the index in valid_mask
                    # We need to map back to the original sorted array
                    valid_indices = np.where(valid_mask)[0]
                    split_idx = valid_indices[max_gain_idx]
                    
                    # Threshold is average of value at split_idx and split_idx+1
                    best_threshold = (sorted_feature_values[split_idx] + sorted_feature_values[split_idx + 1]) / 2.0
                    
        return best_feature, best_threshold, best_gain

    def _get_feature_indices(self, n_features: int) -> np.ndarray:
        """Get indices of features to consider for splitting."""
        if self.max_features is None:
            return np.arange(n_features)
        elif isinstance(self.max_features, int):
            n_features_to_select = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_features_to_select = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            n_features_to_select = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_features_to_select = max(1, int(np.log2(n_features)))
        else:
            return np.arange(n_features)
        
        return self.rng.choice(n_features, n_features_to_select, replace=False)
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        
        return gini
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Avoid log(0)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def _most_common_label(self, y: np.ndarray) -> Any:
        """Return most common label in y."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x: np.ndarray, node: DecisionTreeNode) -> Any:
        """Traverse tree to make prediction for single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def _traverse_tree_to_leaf(self, x: np.ndarray, node: DecisionTreeNode) -> DecisionTreeNode:
        """Traverse tree to leaf node for single sample."""
        if node.is_leaf():
            return node
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree_to_leaf(x, node.left)
        else:
            return self._traverse_tree_to_leaf(x, node.right)
    
    def get_depth(self) -> int:
        """Get depth of the tree."""
        return self._get_node_depth(self.root)
    
    def _get_node_depth(self, node: Optional[DecisionTreeNode]) -> int:
        """Recursively calculate depth of subtree."""
        if node is None or node.is_leaf():
            return 0
        
        left_depth = self._get_node_depth(node.left)
        right_depth = self._get_node_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self) -> int:
        """Get number of leaf nodes in tree."""
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node: Optional[DecisionTreeNode]) -> int:
        """Recursively count leaf nodes."""
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        
        return self._count_leaves(node.left) + self._count_leaves(node.right)

import numpy as np

def gini_index(groups, classes):
    """Compute Gini index for a split based on class distribution.

    Args:
        groups (list or nd-array): Contains the 2 groups resulting from the split.
        classes (list or nd-array): Contains the class values present in these groups.

    Returns:
        float: Gini index (cost) for the given split.
    """
    n_instances = sum([len(group) for group in groups])
    gini = 0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0 
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size 
            score += p * p 
        gini += (1 - score) * (size / n_instances)

    return gini


def test_split(index, value, dataset):
    """Split the dataset into two groups based on a feature value.

    Args:
        index (int): Index of the feature used to split the data.
        value (float): Value of the feature for splitting.
        dataset (list): The dataset to split.

    Returns:
        tuple: Two lists representing the left and right splits.
    """
    left, right = list(), list()
    for row in dataset: 
        curr_val = row[index] 
        if curr_val < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split(bootstrap, n_features):
    """Find the best feature and value to split the dataset.

    Args:
        bootstrap (list): Subsampled dataset (bootstrap sample).
        n_features (int): Number of random features to consider for the split.

    Returns:
        dict: Contains the index, value, and groups for the best split.
    """
    class_val = list(set([row[-1] for row in bootstrap]))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list() 
    while len(features) < n_features:
        index = np.random.randint(len(bootstrap[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features: 
        for row in bootstrap:
            groups = test_split(index, row[index], bootstrap)
            score = gini_index(groups, class_val)
            if score < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], score, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


def to_terminal(group):
    """Return the most common class label in a group (leaf node).

    Args:
        group (list): A group of data points.

    Returns:
        int: Most common class label.
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth=1):
    """Recursively split the node to build a decision tree.

    Args:
        node (dict): The current node to split.
        max_depth (int): Maximum depth of the tree.
        min_size (int): Minimum number of samples required to split a node.
        n_features (int): Number of random features to consider.
        depth (int): Current depth of the tree.

    Returns:
        None: Modifies the node by splitting it into sub-nodes.
    """
    left, right = node["groups"] 
    del(node["groups"])
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return 
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return 
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    elif len(set([row[-1] for row in left])) == 1:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left, n_features)
        split(node["left"], max_depth, min_size, n_features, depth + 1)
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    elif len(set([row[-1] for row in right])) == 1:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right, n_features)
        split(node["right"], max_depth, min_size, n_features, depth + 1)


def build_tree(dataset, max_depth, min_size, n_features):
    """Build a decision tree.

    Args:
        dataset (list): The dataset used to build the tree.
        max_depth (int): Maximum depth of the tree.
        min_size (int): Minimum number of samples required to split a node.
        n_features (int): Number of random features to consider.

    Returns:
        dict: The root node of the decision tree.
    """
    root = get_split(dataset, n_features)
    split(root, max_depth, min_size, n_features)
    return root


def predict(root, row):
    """Make a prediction for a given data point using the decision tree.

    Args:
        root (dict): The root node of the decision tree.
        row (list): A data point.

    Returns:
        int: Predicted class label.
    """
    if row[root["index"]] < root["value"]:
        if isinstance(root["left"], dict):
            return predict(root["left"], row)
        else:
            return root["left"]
    else:
        if isinstance(root["right"], dict):
            return predict(root["right"], row)
        else:
            return root["right"]


def subsample(dataset, ratio):
    """Generate a subsample of the dataset with replacement.

    Args:
        dataset (list): The original dataset.
        ratio (float): Ratio of the dataset size to sample.

    Returns:
        list: Subsample of the dataset.
    """
    sample = list() 
    sample_size = round(len(dataset) * ratio)
    while len(sample) < sample_size:
        index = np.random.randint(len(dataset))
        sample.append(dataset[index])
    return sample


def build_random_forest(train, max_depth, min_size, sample_ratio, n_trees, n_features):
    """Build a random forest model.

    Args:
        train (list): Training dataset.
        max_depth (int): Maximum depth of the trees.
        min_size (int): Minimum number of samples required to split a node.
        sample_ratio (float): Ratio of the dataset size for bootstrapping.
        n_trees (int): Number of trees in the forest.
        n_features (int): Number of random features to consider.

    Returns:
        list: List of decision trees (random forest model).
    """
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_ratio)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    return trees


def bagging_predict(trees, row):
    """Make a prediction for a data point using multiple trees (bagging).

    Args:
        trees (list): List of decision trees.
        row (list): A data point.

    Returns:
        int: Predicted class label based on majority vote.
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest_predict(model, test):
    """Make predictions for a dataset using the random forest model.

    Args:
        model (list): Random forest model (list of decision trees).
        test (list): Dataset for which to make predictions.

    Returns:
        list: Predicted class labels for each data point in the test set.
    """
    predictions = [bagging_predict(model, row) for row in test]
    return predictions

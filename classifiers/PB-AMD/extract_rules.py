from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets, ensemble
import numpy as np


def print_decision_rules(rf):

    for tree_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1  # no support for multi-output

        print('TREE: {}'.format(tree_idx))

        iterator = enumerate(zip(
            tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
        for node_idx, data in iterator:
            left, right, feature, th, value = data

            # left: index of left child (if any)
            # right: index of right child (if any)
            # feature: index of the feature to check
            # th: the threshold to compare against
            # value: values associated with classes

            # for classifier, value is 0 except the index of the class to return
            class_idx = np.argmax(value[0])

            if left == -1 and right == -1:
                print('{} LEAF: return class={}'.format(node_idx, class_idx))
            else:
                print('{} NODE: if feature[{}] < {} then next={} else next={}'.format(
                    node_idx, feature, th, left, right))


def extract_rules_from_tree_to_code(tree, feature_names):
    # Thanks to https://mljar.com/blog/extract-rules-decision-tree/
    f = open("rules.py", "a+")
    for tree_idx, est in enumerate(tree.estimators_):
        tree_ = est.tree_
        feature_names = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
        f.write("def predict({}):".format(", ".join(feature_names)))
        def recurse(node, depth):
            indent = "    " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[node]
                threshold = tree_.threshold[node]
                f.write("{}if {} <= {}:".format(
                    indent, name, np.round(threshold, 2)))
                recurse(tree_.children_left[node], depth + 1)
                f.write("{}else:  # if {} > {}".format(
                    indent, name, np.round(threshold, 2)))
                recurse(tree_.children_right[node], depth + 1)
            else:
                f.write("{}return {}".format(indent, tree_.value[node]))

        recurse(0, 1)

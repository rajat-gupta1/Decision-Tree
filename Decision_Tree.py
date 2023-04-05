import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class Node(object):
    def __init__(self):
        # 0 or 1, basically left child or right child
        self.name = None
        
        # leaf node or not
        self.node_type = None
        
        # Prediction of the leaf node
        self.predicted_class = None
        
        # Values of X in this node
        self.X = None
        
        # The attribute name
        self.test_attribute = None
        
        # The value returned by the attribute
        self.test_value = None
        
        # The list of children of the current node
        self.children = []

    def __repr__(self):

        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {self.X.shape[0]} examples, "
                 f"tests attribute {self.test_attribute} at {self.test_value}")
           
        else:
            s = (f"{self.name} Leaf with {self.X.shape[0]} examples, predicts"
                 f" {self.predicted_class}")
        return s
    

class DecisionTree(object):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        '''
        Fit a tree on data, in which X is a 2-d numpy array
        of inputs, and y is a 1-d numpy array of outputs.
        '''
        self.root = self.recursive_build_tree(
            X, y, curr_depth=0, name='0')
            
    def recursive_build_tree(self, X, y, curr_depth, name):
        """
        Function to build the dtree recursively
        """

        node = Node()
        node.X, node.name = X, name
        
        # If we have reached the max depth or all y are the same or all x are the same
        if curr_depth == self.max_depth or len(np.unique(y)) == 1 or np.isclose(X, X[0]).all():
            node.node_type = 'leaf'
            if len(y) !=0:

                # Predicting class basis the highest occurence in the given examples
                node.predicted_class = self.class_predict(y)

        else:
            # Finding the most import attribute
            A, val = self.Importance(X, y)
            node.test_attribute, node.test_value = A, val

            # Updating the children of the node
            node = self.node_children(X, y, A, val, curr_depth, name, node)

        return node
    
    def node_children(self, X, y, A, val, curr_depth, name, node):
        """
        Function to find the children of the given node
        """

        new_X1 = []
        new_Y1 = []
        new_X2 = []
        new_Y2 = []
        n = len(y)

        for i in range(n):

            # Checking for left/right child
            if X[i][A] < val:
                new_X1.append(X[i, :])
                new_Y1.append(y[i])
            else:
                new_X2.append(X[i, :])
                new_Y2.append(y[i])

        # Recursively calling left/right node
        node_left = self.recursive_build_tree(np.array(new_X1), np.array(new_Y1), curr_depth + 1, name + '.0')
        node_right = self.recursive_build_tree(np.array(new_X2), np.array(new_Y2), curr_depth + 1, name + '.1')
        
        # In case any child didnt have any example of its own
        if node_left.predicted_class == None and node_left.node_type == 'leaf':
            node_left.predicted_class = self.class_predict(y)
        elif node_right.predicted_class == None and node_right.node_type == 'leaf':
            node_right.predicted_class = self.class_predict(y)
        
        node.children.append(node_left)
        node.children.append(node_right)
        return node

    def class_predict(self, y):
        """
        Function to predict the class basis highest occurence in the examples
        """
        values, counts = np.unique(y, return_counts = True)
        return values[counts == counts.max()][0]
        
    def Importance(self, X, y):
        """
        Function to find the most important attribute basis the Entropy
        """
        min_entropy = float('inf')
        min_i = -1
        for i in range(X.shape[1]):

            # Finding the least entropy and the value at which least entropy is found for all examples in a given attribute
            this_entropy, val = self.Importance_cont(X[:, i], y)

            # Checking if entropy for this attribute is lower than the previous least entropy
            if this_entropy < min_entropy:
                min_entropy = this_entropy
                min_i = i
                min_val = val
        return min_i, min_val
    
    def Importance_cont(self, X, y):
        """
        This function extend the Importance function 
        It goes over all values in an attribute and finds the value for which least entropy is obtained
        """

        p = X.argsort()

        # curr_X and curr_Y will arrange X and Y in ascending order of X
        curr_X = X[p]
        curr_Y = y[p]
        n = len(y)
        min_entropy = float('inf')
        val = n - 1
        for i in range(n - 1):

            # The min entropy will be found when adjacent values of X are different
            if curr_X[i] == curr_X[i + 1]:
                continue

            # Finding entropy and normalising it
            this_entropy = (i + 1) / n * self.entropy(curr_Y[:i+1]) + (n - i - 1) / n * self.entropy(curr_Y[i+1:])
            if this_entropy < min_entropy:
                min_entropy = this_entropy
                val = (curr_X[i] + curr_X[i + 1]) / 2
        return min_entropy, val
    
    def predict(self, testset):
        """
        Function to predict the values of Y for a given testset
        """

        y = np.zeros(testset.shape[0])
        for i in range(testset.shape[0]):
            this_node = self.root
            while(this_node.node_type != 'leaf'):

                # Condition for left child
                if testset[i][this_node.test_attribute] < this_node.test_value:
                    this_node = this_node.children[0]

                # Right child
                else:
                    this_node = this_node.children[1]

            # y[i] will be what this leaf is predicting
            y[i] = this_node.predicted_class
        return y
        

    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)
            
    def entropy(self, y):
        'Return the information entropy in 1-d array y'
        
        _, counts = np.unique(y, return_counts = True)
        probs = counts/counts.sum()
        return -(np.log2(probs) * probs).sum()
    
def missing_value_treatment(df):
    """
    This function replaces the missing values in the dataset with the 
    mode value
    """
    for i in range(280):
        if df[i].isnull().sum() > 0:
            df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)

def random_split(df):
    """
    This function randomises the dataset and creates  and returns 
    three different splits of the datasets
    """
    df = df.sample(frac = 1)
    n = len(df.index)
    df1 = df.iloc[:int(1/3*n)]
    df2 = df.iloc[int(1/3*n):int(2/3*n)]
    df3 = df.iloc[int(2/3*n):]
    return df1, df2, df3

def train_test_generation(df1, df2, df3):
    """
    This function generates train and test datasets basis three splits 
    of the datasets given
    """
    X_train = pd.concat([df1.iloc[:, :-1], df2.iloc[:, :-1]])
    Y_train = pd.concat([df1.iloc[:, -1], df2.iloc[:, -1]])
    X_test = df3.iloc[:, :-1]
    Y_test = df3.iloc[:, -1]
    return X_train, Y_train, X_test, Y_test

def accuracy(match1, total, yhat, y):
    """
    The function extends train_test_accuracy function and calculates the
    total matches and total counts in the datasets to be compared
    """

    for j in range(len(yhat)):
        if yhat[j] == y.values[j]:
            match1 += 1
        total += 1
    return match1, total

def train_test_accuracy(df1, df2, df3):
    """
    This function returns the train accuracy and test accuracy after 
    training and doing 3-fold cross validation on the dataset
    """

    Train_accuracy = []
    Test_accuracy = []

    for i in range(2, 17, 2):
        match_train = 0
        total_train = 0
        match_test = 0
        total_test = 0
        for k in range(3):

            # Generating different combination for 3fold cross validation
            if k == 0:
                X_train, Y_train, X_test, Y_test = train_test_generation(df1, df2, df3)
            elif k == 1:
                X_train, Y_train, X_test, Y_test = train_test_generation(df3, df1, df2)
            else:
                X_train, Y_train, X_test, Y_test = train_test_generation(df2, df3, df1)

            tree = DecisionTree(i)

            # Fitting the tree and finding y_hat for train and test datasets
            tree.fit(X_train.values, Y_train.values)
            yhat_train = tree.predict(X_train.values)
            yhat_test = tree.predict(X_test.values)

            # Updating the matching values and the total values for train and test
            match_train, total_train = accuracy(match_train, total_train, yhat_train, Y_train)
            match_test, total_test = accuracy(match_test, total_test, yhat_test, Y_test)

        Train_accuracy.append(match_train/total_train)
        Test_accuracy.append(match_test/total_test)
    return Train_accuracy, Test_accuracy

def plot_generation(Train_accuracy, Test_accuracy):
    """
    The function generates a plot and saves it as a pdf in the current directory
    """

    # The x axis
    data_points = [i for i in range(2, 17, 2)]

    # Plotting both train and test accuracies on the y axis
    plt.plot(data_points, Train_accuracy, 'r', label = "Train accuracy")
    plt.plot(data_points, Test_accuracy, 'b', label = "Test accuracy")
    plt.xlabel("Depth of the tree")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig("validation.png", format="png", bbox_inches="tight")

def main(argv): 
    
    # Reading file using the first argument of the command line
    df = pd.read_csv(sys.argv[1], header=None, na_values="?")
    missing_value_treatment(df)
    df1, df2, df3 = random_split(df)
    Train_accuracy, Test_accuracy = train_test_accuracy(df1, df2, df3)
    plot_generation(Train_accuracy, Test_accuracy)


if __name__=="__main__":
    main(sys.argv[1:])
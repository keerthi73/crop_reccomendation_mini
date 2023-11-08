from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pandas as pd

# Load your dataset
dataset = pd.read_csv('cropdata.csv')  # Replace 'your_dataset.csv' with your dataset file path

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Create and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Define feature names (replace with your feature names) and class names (extracted from dataset)
feature_names = ['temperature', 'humidity', 'ph', 'rainfall']
class_names = list(dataset['label'].unique())

# Export the decision tree to a DOT file (Graphviz format)
dot_data = export_graphviz(
    clf,
    out_file=None,  # Use None to return the DOT data as a string
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
)

# Create a Graphviz object from the DOT data
graph = graphviz.Source(dot_data)

# Save the decision tree as an image (optional)
graph.render("decision_tree")  # This will create a file named "decision_tree.pdf" in the current directory

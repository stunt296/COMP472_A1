import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Read the CSV files
penguin_data = pd.read_csv('penguins.csv')
abalone_data = pd.read_csv('abalone.csv')

# Convert the string values to numerical
penguin_data['island'] = penguin_data['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
penguin_data['sex'] = penguin_data['sex'].map({'MALE': 0, 'FEMALE': 1})

# Create the png files
class_distribution = penguin_data['species'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%')
plt.savefig('penguin-classes.png')
plt.show()

class_distribution = abalone_data['Type'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%')
plt.savefig('abalone-classes.png')
plt.show()

# Split the datasets
X_penguin = penguin_data.drop('species', axis=1)
y_penguin = penguin_data['species']
X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin, test_size=0.2, random_state=42)

X_abalone = abalone_data.drop('Type', axis=1)
y_abalone = abalone_data['Type']
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone, test_size=0.2, random_state=42)


"""----------------PENGUIN TRAINING--------------------"""

# Create and train the Base-DT with default parameters
base_dt_penguin = DecisionTreeClassifier()
base_dt_penguin.fit(X_train_penguin, y_train_penguin)

# Visualize the Base-DT
plt.figure(figsize=(16, 8))
plot_tree(base_dt_penguin, filled=True, fontsize=10)
plt.show()

# Evaluate the Base-DT
base_dt_penguin_accuracy = base_dt_penguin.score(X_test_penguin, y_test_penguin)
print("Base-DT Accuracy for Penguin:", base_dt_penguin_accuracy)

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, None],  # Choose your depth values
    'min_samples_split': [2, 5, 10],  # Choose your split values
}

# Create a Decision Tree classifier
top_dt_penguin = DecisionTreeClassifier()

# Perform grid search using cross-validation
grid_search_penguin = GridSearchCV(top_dt_penguin, param_grid, cv=5)
grid_search_penguin.fit(X_train_penguin, y_train_penguin)

# Get the best hyperparameters
best_params_penguin = grid_search_penguin.best_params_
print("Best Hyperparameters for Penguin:", best_params_penguin)

# Train the Top-DT with the best hyperparameters
top_dt_penguin = DecisionTreeClassifier(**best_params_penguin)
top_dt_penguin.fit(X_train_penguin, y_train_penguin)

# Visualize the Top-DT
plt.figure(figsize=(16, 8))
plot_tree(top_dt_penguin, filled=True, fontsize=10)
plt.show()

# Evaluate the Top-DT
top_dt_penguin_accuracy = top_dt_penguin.score(X_test_penguin, y_test_penguin)
print("Top-DT Accuracy for Penguin:", top_dt_penguin_accuracy, "\n")


"""----------------ABALONE TRAINING--------------------"""

# Create and train the Base-DT with default parameters
base_dt_abalone = DecisionTreeClassifier()
base_dt_abalone.fit(X_train_abalone, y_train_abalone)

# Visualize the Base-DT
plt.figure(figsize=(16, 8))
plot_tree(base_dt_abalone, filled=True, fontsize=10, max_depth=3)
plt.show()

# Evaluate the Base-DT
base_dt_abalone_accuracy = base_dt_abalone.score(X_test_abalone, y_test_abalone)
print("Base-DT Accuracy for Abalone:", base_dt_abalone_accuracy)

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5, 10],
}

# Create a Decision Tree classifier
top_dt_abalone = DecisionTreeClassifier()

# Perform grid search using cross-validation
grid_search_abalone = GridSearchCV(top_dt_abalone, param_grid, cv=5)
grid_search_abalone.fit(X_train_abalone, y_train_abalone)

# Get the best hyperparameters
best_params_abalone = grid_search_abalone.best_params_
print("Best Hyperparameters for Abalone:", best_params_abalone)

# Train the Top-DT with the best hyperparameters
top_dt_abalone = DecisionTreeClassifier(**best_params_abalone)
top_dt_abalone.fit(X_train_abalone, y_train_abalone)

# Visualize the Top-DT
plt.figure(figsize=(16, 8))
plot_tree(top_dt_abalone, filled=True, fontsize=10, max_depth=3)
plt.show()

# Evaluate the Top-DT
top_dt_accuracy_abalone = top_dt_abalone.score(X_test_abalone, y_test_abalone)
print("Top-DT Accuracy for Abalone:", top_dt_accuracy_abalone)

"""----------------PENGUIN TRAINING - MLP--------------------"""

# Create and train the Base-MLP with default parameters
mlp_p = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', max_iter=1000, random_state=42)
mlp_p.fit(X_train_penguin, y_train_penguin)

# Evaluate the Base-MLP
mlp_p_acc = mlp_p.score(X_test_penguin, y_test_penguin)
print("Base-MLP Accuracy for Penguin:", mlp_p_acc)

# Define hyperparameters to tune
param_grid_mlp = {
    'activation': ['tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd'],
}

# Create an MLP classifier
top_mlp_p = MLPClassifier(max_iter=1000, random_state=42)

# Perform grid search using cross-validation
grid_p = GridSearchCV(top_mlp_p, param_grid_mlp, cv=5)
grid_p.fit(X_train_penguin, y_train_penguin)

# Get the best hyperparameters
best_p = grid_p.best_params_
print("Best Hyperparameters for Top-MLP Penguin:", best_p)

# Train the Top-MLP with the best hyperparameters
top_mlp_p = MLPClassifier(**best_p, max_iter=1000, random_state=42)
top_mlp_p.fit(X_train_penguin, y_train_penguin)

# Evaluate the Top-MLP
top_mpl_p_acc = top_mlp_p.score(X_test_penguin, y_test_penguin)
print("Top-MLP Accuracy for Penguin:", top_mpl_p_acc, "\n")

"""----------------ABALONE TRAINING - MLP--------------------"""

# Create and train the Base-MLP with default parameters
mpl_a = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', max_iter=1000, random_state=42)
mpl_a.fit(X_train_abalone, y_train_abalone)

# Evaluate the Base-MLP
mpl_a_acc = mpl_a.score(X_test_abalone, y_test_abalone)
print("Base-MLP Accuracy for Abalone:", mpl_a_acc)

# Create an MLP classifier
top_mpl_a = MLPClassifier(max_iter=1000, random_state=42)

# Perform grid search using cross-validation
grid_a = GridSearchCV(top_mpl_a, param_grid_mlp, cv=5)
grid_a.fit(X_train_abalone, y_train_abalone)

# Get the best hyperparameters
best_a = grid_a.best_params_
print("Best Hyperparameters for Top-MLP Abalone:", best_a)

# Train the Top-MLP with the best hyperparameters
top_mpl_a = MLPClassifier(**best_a, max_iter=1000, random_state=42)
top_mpl_a.fit(X_train_abalone, y_train_abalone)

# Evaluate the Top-MLP
top_mpl_a_acc = top_mpl_a.score(X_test_abalone, y_test_abalone)
print("Top-MLP Accuracy for Abalone:", top_mpl_a_acc)

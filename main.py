import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
import numpy as np

# Read the CSV files
penguin_data = pd.read_csv("penguins.csv")
abalone_data = pd.read_csv("abalone.csv")

# Convert the string values to numerical
penguin_data["island"] = penguin_data["island"].map(
    {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
)
penguin_data["sex"] = penguin_data["sex"].map({"MALE": 0, "FEMALE": 1})

# Create the png files
class_distribution = penguin_data["species"].value_counts()
class_distribution.plot(kind="pie", autopct="%1.1f%%")
plt.savefig("penguin-classes.png")
plt.show()

class_distribution = abalone_data["Type"].value_counts()
class_distribution.plot(kind="pie", autopct="%1.1f%%")
plt.savefig("abalone-classes.png")
plt.show()

# Split the datasets
X_penguin = penguin_data.drop("species", axis=1)
y_penguin = penguin_data["species"]
X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(
    X_penguin, y_penguin, test_size=0.2, random_state=42
)

X_abalone = abalone_data.drop("Type", axis=1)
y_abalone = abalone_data["Type"]
X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(
    X_abalone, y_abalone, test_size=0.2, random_state=42
)


accuracy_penguin = []
accuracy_abalone = []
macro_f1_penguin = []
macro_f1_abalone = []
weighted_f1_penguin = []
weighted_f1_abalone = []

for _ in range(2):
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
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, None],  # Choose your depth values
        "min_samples_split": [2, 5, 10],  # Choose your split values
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

    # ... (Rest of the existing code)

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
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5, 10],
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
    mlp_p = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        activation="logistic",
        max_iter=1000,
        random_state=42,
    )
    mlp_p.fit(X_train_penguin, y_train_penguin)

    # Evaluate the Base-MLP
    mlp_p_acc = mlp_p.score(X_test_penguin, y_test_penguin)
    print("Base-MLP Accuracy for Penguin:", mlp_p_acc)

    # Define hyperparameters to tune
    param_grid_mlp = {
        "activation": ["tanh", "relu"],
        "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
        "solver": ["adam", "sgd"],
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
    mpl_a = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        activation="logistic",
        max_iter=1000,
        random_state=42,
    )
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

    """----------------PENGUIN LOGS--------------------"""

    with open("penguin-performance.txt", "a") as f:
        f.write("Base Decision Tree for Penguins\n")
        f.write("---------------------------------\n")

        f.write(
            "(A) Best Hyperparameters for Penguin: " + str(best_params_penguin) + "\n"
        )
        f.write("Best Hyperparameters for Top-MLP Penguin:" + str(best_p) + "\n")
        f.write("---------------------------------\n")

        # Confusion Matrix
        f.write("(B) Confusion Matrix:\n")
        f.write("Base-DT \n")
        f.write(
            str(
                confusion_matrix(
                    y_test_penguin, base_dt_penguin.predict(X_test_penguin)
                )
            )
            + "\n"
        )
        f.write("Top-DT \n")
        f.write(
            str(
                confusion_matrix(y_test_penguin, top_dt_penguin.predict(X_test_penguin))
            )
            + "\n"
        )
        f.write("Base-MLP \n")
        f.write(
            str(confusion_matrix(y_test_penguin, mlp_p.predict(X_test_penguin))) + "\n"
        )
        f.write("Top-MLP \n")
        f.write(
            str(confusion_matrix(y_test_penguin, top_mlp_p.predict(X_test_penguin)))
            + "\n"
        )
        f.write("---------------------------------\n")

        # Classification Report
        f.write("(C) Classification Report:\n")
        f.write("Base-DT\n")
        f.write(
            classification_report(
                y_test_penguin, base_dt_penguin.predict(X_test_penguin)
            )
            + "\n"
        )
        f.write("Top-DT\n")
        f.write(
            classification_report(
                y_test_penguin, top_dt_penguin.predict(X_test_penguin)
            )
            + "\n"
        )
        f.write("Base-MLP\n")
        f.write(
            classification_report(y_test_penguin, mlp_p.predict(X_test_penguin)) + "\n"
        )
        f.write("Top-MLP\n")
        f.write(
            classification_report(y_test_penguin, top_mlp_p.predict(X_test_penguin))
            + "\n"
        )
        f.write("---------------------------------\n")

        # Accuracy, Macro-average F1, and Weighted-average F1
        f.write("(D) Accuracy, Macro-average F1, and Weighted-average F1:\n")
        f.write(
            "Accuracy for Base-DT: {:.4f}\n".format(
                base_dt_penguin.score(X_test_penguin, y_test_penguin)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Top-DT: {:.4f}\n".format(
                top_dt_penguin.score(X_test_penguin, y_test_penguin)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Base-MLP: {:.4f}\n".format(
                mlp_p.score(X_test_penguin, y_test_penguin)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Top-MLP: {:.4f}\n".format(
                top_mlp_p.score(X_test_penguin, y_test_penguin)
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Top-DT: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    top_dt_penguin.predict(X_test_penguin),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Base-DT: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    base_dt_penguin.predict(X_test_penguin),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Base-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    mlp_p.predict(X_test_penguin),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Macro-average F1 for Top-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    top_mlp_p.predict(X_test_penguin),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Base-DT: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    base_dt_penguin.predict(X_test_penguin),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Top-DT: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    top_dt_penguin.predict(X_test_penguin),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Base-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    mlp_p.predict(X_test_penguin),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Top-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_penguin,
                    top_mlp_p.predict(X_test_penguin),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        """----------------ABALONE LOGS--------------------"""

    with open("abalone-performance.txt", "a") as f:
        f.write("Base Decision Tree for Abalone\n")
        f.write("---------------------------------\n")

        f.write(
            "(A) Best Hyperparameters for Abalone: " + str(best_params_abalone) + "\n"
        )
        f.write("Best Hyperparameters for Top-MLP Abalone:" + str(best_a) + "\n")
        f.write("---------------------------------\n")

        # Confusion Matrix
        f.write("(B) Confusion Matrix:\n")
        f.write("Base-DT : \n")
        f.write(
            str(
                confusion_matrix(
                    y_test_abalone, base_dt_abalone.predict(X_test_abalone)
                )
            )
            + "\n"
        )
        f.write("Top-DT : \n")
        f.write(
            str(
                confusion_matrix(y_test_abalone, top_dt_abalone.predict(X_test_abalone))
            )
            + "\n"
        )
        f.write("Base-MLP : \n")
        f.write(
            str(confusion_matrix(y_test_abalone, mpl_a.predict(X_test_abalone))) + "\n"
        )
        f.write("Top-MLP : \n")
        f.write(
            str(confusion_matrix(y_test_abalone, top_mpl_a.predict(X_test_abalone)))
            + "\n"
        )
        f.write("---------------------------------\n")

        # Classification Report
        f.write("(C) Classification Report:\n")
        f.write("Base-DT : \n")
        f.write(
            classification_report(
                y_test_abalone, base_dt_abalone.predict(X_test_abalone)
            )
            + "\n"
        )
        f.write("Top-DT : \n")
        f.write(
            classification_report(
                y_test_abalone, top_dt_abalone.predict(X_test_abalone)
            )
            + "\n"
        )
        f.write("Base-MLP : \n")
        f.write(
            classification_report(y_test_abalone, mpl_a.predict(X_test_abalone)) + "\n"
        )
        f.write("Top-MLP : \n")
        f.write(
            classification_report(y_test_abalone, top_mpl_a.predict(X_test_abalone))
            + "\n"
        )
        f.write("---------------------------------\n")

        # Accuracy, Macro-average F1, and Weighted-average F1
        f.write("(D) Accuracy, Macro-average F1, and Weighted-average F1:\n")
        f.write(
            "Accuracy for Base-DT: {:.4f}\n".format(
                base_dt_abalone.score(X_test_abalone, y_test_abalone)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Top-DT: {:.4f}\n".format(
                top_dt_abalone.score(X_test_abalone, y_test_abalone)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Base-MLP: {:.4f}\n".format(
                mpl_a.score(X_test_abalone, y_test_abalone)
            )
            + "\n\n"
        )
        f.write(
            "Accuracy for Top-MLP: {:.4f}\n".format(
                top_mpl_a.score(X_test_abalone, y_test_abalone)
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Top-DT: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    top_dt_abalone.predict(X_test_abalone),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Base-DT: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    base_dt_abalone.predict(X_test_abalone),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )
        f.write(
            "Macro-average F1 for Base-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    mpl_a.predict(X_test_abalone),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Macro-average F1 for Top-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    top_mpl_a.predict(X_test_abalone),
                    average="macro",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Base-DT: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    base_dt_abalone.predict(X_test_abalone),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Top-DT: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    top_dt_abalone.predict(X_test_abalone),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Base-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    mpl_a.predict(X_test_abalone),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

        f.write(
            "Weighted-average F1 for Top-MLP: {:.4f}\n".format(
                f1_score(
                    y_test_abalone,
                    top_mpl_a.predict(X_test_abalone),
                    average="weighted",
                    zero_division=0,
                )
            )
            + "\n\n"
        )

    accuracy_penguin.append(base_dt_penguin.score(X_test_penguin, y_test_penguin))
    accuracy_abalone.append(base_dt_abalone.score(X_test_abalone, y_test_abalone))

    macro_f1_penguin.append(
        f1_score(
            y_test_penguin,
            base_dt_penguin.predict(X_test_penguin),
            average="macro",
            zero_division=0,
        )
    )
    macro_f1_abalone.append(
        f1_score(
            y_test_abalone,
            base_dt_abalone.predict(X_test_abalone),
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1_penguin.append(
        f1_score(
            y_test_penguin,
            base_dt_penguin.predict(X_test_penguin),
            average="weighted",
            zero_division=0,
        )
    )

    weighted_f1_abalone.append(
        f1_score(
            y_test_abalone,
            base_dt_abalone.predict(X_test_abalone),
            average="weighted",
            zero_division=0,
        )
    )

with open("penguin-performance.txt", "a") as f:
    f.write("(A) Average Accuracy:" + str(np.mean(accuracy_penguin)) + "\n")
    f.write("(A) Variance Accuracy:" + str(np.var(accuracy_penguin)) + "\n")
    f.write("(B) Average Macro-F1:" + str(np.mean(macro_f1_penguin)) + "\n")
    f.write("(B) Variance Macro-F1:" + str(np.var(macro_f1_penguin)) + "\n")
    f.write("(C) Average Weighted-F1 :" + str(np.mean(weighted_f1_penguin)) + "\n")
    f.write("(C) Variance Weighted-F1 :" + str(np.var(weighted_f1_penguin)) + "\n")

with open("abalone-performance.txt", "a") as f:
    f.write("(A) Average Accuracy:" + str(np.mean(accuracy_abalone)) + "\n")
    f.write("(A) Variance Accuracy:" + str(np.var(accuracy_abalone)) + "\n")
    f.write("(B) Average Macro-F1:" + str(np.mean(macro_f1_abalone)) + "\n")
    f.write("(B) Variance Macro-F1:" + str(np.var(macro_f1_abalone)) + "\n")
    f.write("(C) Average Weighted-F1 :" + str(np.mean(weighted_f1_abalone)) + "\n")
    f.write("(C) Variance Weighted-F1 :" + str(np.var(weighted_f1_abalone)) + "\n")

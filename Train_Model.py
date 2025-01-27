import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

# Generate example data
np.random.seed(42)
X = np.random.normal(size=(200, 10))  # 200 samples, 10 features
Y = X[:, :2] + np.random.normal(size=(200, 2))  # Linear relationship + noise

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Create and train PLS model
pls = PLSRegression(n_components=2)
pls.fit(X_train, Y_train)

# Calculate scores for training and testing sets
X_train_scores, Y_train_scores = pls.transform(X_train, Y_train)
X_test_scores, Y_test_scores = pls.transform(X_test, Y_test)

# Create plots
plt.figure(figsize=(12, 10))

# Plot 1: Component 1 X vs Component 1 Y
plt.subplot(2, 2, 1)
plt.scatter(X_train_scores[:, 0], Y_train_scores[:, 0], label="train", alpha=0.7)
plt.scatter(X_test_scores[:, 0], Y_test_scores[:, 0], label="test", alpha=0.7)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title("Comp. 1: X vs Y (test corr = {:.2f})".format(np.corrcoef(X_test_scores[:, 0], Y_test_scores[:, 0])[0, 1]))
plt.legend()

# Plot 2: Component 1 X vs Component 2 X
plt.subplot(2, 2, 2)
plt.scatter(X_train_scores[:, 0], X_train_scores[:, 1], label="train", alpha=0.7)
plt.scatter(X_test_scores[:, 0], X_test_scores[:, 1], label="test", alpha=0.7)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title("X comp. 1 vs X comp. 2 (test corr = {:.2f})".format(np.corrcoef(X_test_scores[:, 0], X_test_scores[:, 1])[0, 1]))
plt.legend()

# Plot 3: Component 1 Y vs Component 2 Y
plt.subplot(2, 2, 3)
plt.scatter(Y_train_scores[:, 0], Y_train_scores[:, 1], label="train", alpha=0.7)
plt.scatter(Y_test_scores[:, 0], Y_test_scores[:, 1], label="test", alpha=0.7)
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.title("Y comp. 1 vs Y comp. 2, (test corr = {:.2f})".format(np.corrcoef(Y_test_scores[:, 0], Y_test_scores[:, 1])[0, 1]))
plt.legend()

# Plot 4: Component 2 X vs Component 2 Y
plt.subplot(2, 2, 4)
plt.scatter(X_train_scores[:, 1], Y_train_scores[:, 1], label="train", alpha=0.7)
plt.scatter(X_test_scores[:, 1], Y_test_scores[:, 1], label="test", alpha=0.7)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title("Comp. 2: X vs Y (test corr = {:.2f})".format(np.corrcoef(X_test_scores[:, 1], Y_test_scores[:, 1])[0, 1]))
plt.legend()

plt.tight_layout()
plt.show()

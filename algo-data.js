const ALGO_DATA = {
  "linear-regression": {
    name: "Linear Regression",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "The foundation of predictive modeling — fitting a straight line through data to predict continuous outcomes.",
    steps: [
      "Collect labeled data with input features (X) and a continuous target (y).",
      "Assume a linear relationship: y = w₁x₁ + w₂x₂ + ... + b (weights + bias).",
      "Define a loss function — Mean Squared Error (MSE) measures how far predictions are from actual values.",
      "Use gradient descent (or normal equation) to find weights that minimize MSE.",
      "The fitted line can now predict y for any new input X."
    ],
    formula: "y = Xw + b\n\nMSE = (1/n) Σ(yᵢ - ŷᵢ)²\n\nNormal equation: w = (XᵀX)⁻¹Xᵀy",
    formulaNote: "w = weight vector, b = bias, ŷ = predicted value, n = number of samples",
    example: {
      title: "Predicting House Prices",
      desc: "A real estate company wants to estimate house prices. Features include square footage (1,500 sq ft), number of bedrooms (3), and distance to city center (5 miles). The model learns: Price = $150 × sqft + $20,000 × bedrooms - $5,000 × distance + $50,000. For a 1,500 sq ft, 3-bed, 5-mile house: $150 × 1500 + $20,000 × 3 + (-$5,000 × 5) + $50,000 = $310,000."
    },
    code: `import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1500, 3, 5], [2000, 4, 3], [1200, 2, 8]])  # sqft, beds, distance
y = np.array([310000, 450000, 220000])  # prices

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict new house
new_house = np.array([[1800, 3, 4]])
predicted_price = model.predict(new_house)
print(f"Predicted price: \${predicted_price[0]:,.0f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.0f}")`,
    pros: ["Simple and fast to train", "Highly interpretable — each weight shows feature importance", "Works well when relationships are truly linear", "No hyperparameters to tune (basic form)", "Great baseline model to start with"],
    cons: ["Cannot capture non-linear relationships", "Sensitive to outliers", "Assumes features are independent (multicollinearity issues)", "Underfits complex data"],
    complexity: { training: "O(n·d²) or O(d³)", prediction: "O(d)", space: "O(d)" }
  },

  "logistic-regression": {
    name: "Logistic Regression",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "A classification algorithm that outputs probabilities using the sigmoid function — despite its misleading name.",
    steps: [
      "Compute a weighted sum of inputs: z = w₁x₁ + w₂x₂ + ... + b.",
      "Pass z through the sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ) to get a probability between 0 and 1.",
      "Apply a threshold (usually 0.5) to convert probability to class: ≥ 0.5 → class 1, < 0.5 → class 0.",
      "Use Binary Cross-Entropy as the loss function to measure prediction quality.",
      "Optimize weights via gradient descent to minimize the loss."
    ],
    formula: "P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))\n\nLoss = -1/n Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]",
    formulaNote: "σ = sigmoid function, outputs probability between 0 and 1",
    example: {
      title: "Email Spam Detection",
      desc: "An email service classifies incoming mail. Features: number of exclamation marks (5), contains 'free' (1=yes), sender in contacts (0=no), link count (8). Model computes z = 0.3×5 + 2.1×1 + (-3.0)×0 + 0.4×8 = 6.8. Sigmoid(6.8) = 0.999 → 99.9% probability of spam. Since 0.999 > 0.5, the email is classified as spam."
    },
    code: `from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Features: [exclamation_marks, has_free, in_contacts, link_count]
X = [[5,1,0,8], [0,0,1,1], [3,1,0,5], [1,0,1,0], [7,1,0,12], [0,0,1,2]]
y = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

model = LogisticRegression()
model.fit(X, y)

new_email = [[4, 1, 0, 6]]
prob = model.predict_proba(new_email)[0]
print(f"Spam probability: {prob[1]:.1%}")
print(f"Classification: {'SPAM' if prob[1] > 0.5 else 'NOT SPAM'}")`,
    pros: ["Outputs calibrated probabilities, not just classes", "Very fast to train and predict", "Highly interpretable with odds ratios", "Works well with linearly separable data", "Low risk of overfitting with regularization"],
    cons: ["Assumes linear decision boundary", "Cannot solve XOR or complex non-linear problems", "Requires feature engineering for non-linear relationships", "Sensitive to class imbalance"],
    complexity: { training: "O(n·d)", prediction: "O(d)", space: "O(d)" }
  },

  "decision-trees": {
    name: "Decision Trees",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "A flowchart-like model that makes decisions by splitting data on feature values — one of the most interpretable algorithms.",
    steps: [
      "Start with all training data at the root node.",
      "Find the best feature and split point that maximizes information gain (or minimizes Gini impurity).",
      "Split data into left/right child nodes based on the condition (e.g., age > 30).",
      "Recursively repeat for each child node until a stopping criterion is met (max depth, min samples, or pure node).",
      "Each leaf node holds a prediction: majority class (classification) or mean value (regression)."
    ],
    formula: "Gini Impurity = 1 - Σ pᵢ²\n\nEntropy = -Σ pᵢ log₂(pᵢ)\n\nInfo Gain = Entropy(parent) - Σ (nⱼ/n) × Entropy(childⱼ)",
    formulaNote: "pᵢ = proportion of class i in a node; lower Gini/Entropy = purer node",
    example: {
      title: "Loan Approval System",
      desc: "A bank builds a decision tree for loan approval. The tree learns: First, check income > $50k? If yes, check credit score > 700? If yes → Approve. If no, check existing debt < $10k? If yes → Approve with conditions. If no → Deny. Each path through the tree is a transparent, explainable rule that loan officers can verify and audit."
    },
    code: `from sklearn.tree import DecisionTreeClassifier, export_text

# Features: [income_k, credit_score, debt_k, years_employed]
X = [[80,750,5,10], [30,600,15,2], [55,720,8,5],
     [90,680,3,15], [25,580,20,1], [60,710,12,7]]
y = [1, 0, 1, 1, 0, 0]  # 1=approve, 0=deny

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Print the decision rules
print(export_text(tree, feature_names=["income","credit","debt","years"]))

# Predict new applicant
applicant = [[65, 700, 7, 6]]
print(f"Decision: {'APPROVED' if tree.predict(applicant)[0] else 'DENIED'}")`,
    pros: ["Extremely interpretable — you can visualize the full tree", "Handles both numerical and categorical data", "No feature scaling required", "Captures non-linear relationships", "Fast training and prediction"],
    cons: ["Prone to overfitting (high variance)", "Unstable — small data changes can create entirely different trees", "Biased toward features with many levels", "Greedy splits aren't globally optimal"],
    complexity: { training: "O(n·d·log n)", prediction: "O(log n)", space: "O(nodes)" }
  },

  "random-forest": {
    name: "Random Forest",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "An ensemble of decision trees that reduces overfitting through bagging and random feature selection.",
    steps: [
      "Create B bootstrap samples (random samples with replacement) from the training data.",
      "Train one decision tree on each bootstrap sample, but at each split only consider a random subset of features (√d for classification, d/3 for regression).",
      "Each tree grows fully (no pruning), deliberately overfitting its bootstrap sample.",
      "For prediction, every tree votes independently.",
      "Aggregate votes: majority vote (classification) or average (regression). Disagreement between trees is what provides robustness."
    ],
    formula: "ŷ = mode(tree₁(x), tree₂(x), ..., treeB(x))    [classification]\nŷ = (1/B) Σ treeᵢ(x)                              [regression]\n\nFeatures per split: m = √d (classification), m = d/3 (regression)",
    formulaNote: "B = number of trees (typically 100-500), d = total features, m = features considered per split",
    example: {
      title: "Credit Card Fraud Detection",
      desc: "A payment processor uses 200 decision trees. Each tree sees a random 63% of transactions (bootstrap) and considers only 4 of 20 features per split (√20 ≈ 4). For a suspicious transaction, 178 trees vote 'fraud' and 22 vote 'legitimate' → 89% fraud confidence. The randomness means no single tree's mistake dominates — the forest is much more robust than any individual tree."
    },
    code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Features: [amount, hour, distance_from_home, is_foreign, ...]
X_train = [[500,14,5,0], [5000,3,200,1], [50,10,2,0],
           [8000,2,500,1], [120,16,10,0], [3000,1,300,1]]
y_train = [0, 1, 0, 1, 0, 1]  # 0=legit, 1=fraud

rf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                            random_state=42)
rf.fit(X_train, y_train)

# Feature importance
for name, imp in zip(["amount","hour","distance","foreign"],
                     rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# Predict
transaction = [[4500, 2, 350, 1]]
print(f"Fraud probability: {rf.predict_proba(transaction)[0][1]:.1%}")`,
    pros: ["Much more accurate than single decision trees", "Resistant to overfitting due to ensemble averaging", "Provides feature importance rankings", "Handles missing values and mixed data types", "Requires minimal hyperparameter tuning"],
    cons: ["Less interpretable than a single decision tree", "Slower to train and predict than simple models", "Can be memory-intensive with many trees", "Struggles with very high-dimensional sparse data"],
    complexity: { training: "O(B·n·d·log n)", prediction: "O(B·log n)", space: "O(B·nodes)" }
  },

  "svm": {
    name: "Support Vector Machine",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "Finds the optimal hyperplane that separates classes with the maximum margin — especially powerful in high-dimensional spaces.",
    steps: [
      "Plot data points in feature space. The goal is to find a separating hyperplane.",
      "Find the hyperplane that maximizes the margin — the distance to the nearest data points (support vectors) of each class.",
      "Only the support vectors (points closest to the boundary) determine the hyperplane. Other points are irrelevant.",
      "For non-linearly separable data, use the kernel trick to project data into a higher-dimensional space where it becomes separable.",
      "Use soft margin (C parameter) to allow some misclassifications when data isn't perfectly separable."
    ],
    formula: "Maximize margin: 2/||w||\nSubject to: yᵢ(wᵀxᵢ + b) ≥ 1\n\nKernel trick: K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)\nRBF kernel: K(x,x') = exp(-γ||x-x'||²)",
    formulaNote: "w = weight vector, support vectors are the critical boundary points, C controls margin softness",
    example: {
      title: "Classifying Cancer from Gene Expression",
      desc: "A research team has 200 tissue samples with 15,000 gene expressions each. SVMs excel here because the feature space (15,000 dimensions) is much larger than the sample count (200). Using an RBF kernel, the SVM finds complex boundaries in gene-expression space. Only ~30 samples become support vectors — the model is efficient and generalizes well even with limited data."
    },
    code: `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Gene expression data (simplified)
X = [[2.1,3.5,1.2], [1.8,4.1,0.9], [5.2,1.1,4.8],
     [4.9,0.8,5.1], [2.5,3.8,1.5], [5.5,1.3,4.5]]
y = [0, 0, 1, 1, 0, 1]  # 0=benign, 1=malignant

# SVM with RBF kernel (scaling is important!)
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
model.fit(X, y)

sample = [[3.5, 2.5, 3.0]]
print(f"Prediction: {'Malignant' if model.predict(sample)[0] else 'Benign'}")
print(f"Support vectors: {model.named_steps['svc'].n_support_}")`,
    pros: ["Excellent in high-dimensional spaces", "Memory-efficient (only stores support vectors)", "Versatile — different kernels for different problems", "Effective when dimensions > samples", "Strong theoretical guarantees"],
    cons: ["Slow on large datasets — O(n²) to O(n³) training", "Doesn't output probabilities natively", "Sensitive to feature scaling", "Kernel and C selection requires tuning", "Hard to interpret in kernel space"],
    complexity: { training: "O(n² · d) to O(n³)", prediction: "O(sv · d)", space: "O(sv · d)" }
  },

  "knn": {
    name: "K-Nearest Neighbors",
    category: "Supervised Learning",
    badge: "supervised",
    subtitle: "The simplest ML algorithm — classify a new point by looking at what its closest neighbors are. No training phase at all.",
    steps: [
      "Store all training data — there is no 'training' step (lazy learning).",
      "When a new point arrives, compute its distance to every stored point (Euclidean, Manhattan, etc.).",
      "Select the K closest neighbors.",
      "For classification: majority vote among K neighbors. For regression: average their values.",
      "Return the prediction. Choosing K matters — small K is noisy, large K is too smooth."
    ],
    formula: "Euclidean distance: d(x,y) = √(Σ(xᵢ - yᵢ)²)\n\nPrediction = mode(labels of K nearest neighbors)\n\nOptimal K: typically √n, always odd to avoid ties",
    formulaNote: "No parameters are learned — the entire training set IS the model",
    example: {
      title: "Movie Recommendation",
      desc: "A streaming service recommends movies. Each movie is represented by features: action_score=8, romance=2, comedy=5, drama=3. To recommend movies for a user who liked Movie A, find the 5 nearest movies in feature space. If 3 of the 5 are sci-fi thrillers and 2 are action comedies, recommend more sci-fi thrillers. The 'neighborhood' captures taste similarity."
    },
    code: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Flowers: [petal_length, petal_width, sepal_length, sepal_width]
X = [[1.4,0.2,5.1,3.5], [4.7,1.4,7.0,3.2], [1.3,0.2,4.6,3.1],
     [5.0,1.5,6.3,3.3], [1.5,0.3,5.0,3.6], [4.5,1.5,6.5,2.8]]
y = ['setosa','versicolor','setosa','versicolor','setosa','versicolor']

# Always scale features for KNN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

new_flower = scaler.transform([[3.0, 0.8, 5.5, 3.0]])
print(f"Prediction: {knn.predict(new_flower)[0]}")
print(f"Neighbor distances: {knn.kneighbors(new_flower)[0][0].round(2)}")`,
    pros: ["Zero training time — instantly adapts to new data", "Intuitive and easy to explain", "No assumptions about data distribution", "Naturally handles multi-class problems", "Non-parametric — fits any decision boundary"],
    cons: ["Slow prediction — must scan entire dataset", "Terrible with high-dimensional data (curse of dimensionality)", "Sensitive to irrelevant features and scale", "Must store entire training set in memory", "Choosing K is tricky"],
    complexity: { training: "O(1) — just store data", prediction: "O(n · d)", space: "O(n · d)" }
  },

  "k-means": {
    name: "K-Means Clustering",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "Partitions data into K groups by iteratively assigning points to the nearest centroid and updating centroids.",
    steps: [
      "Choose K (number of clusters) and randomly initialize K centroid positions.",
      "Assignment step: assign each data point to the nearest centroid (by Euclidean distance).",
      "Update step: recalculate each centroid as the mean of all points assigned to it.",
      "Repeat assignment and update until centroids stop moving (convergence).",
      "The final centroids define K clusters. Use the elbow method or silhouette score to choose optimal K."
    ],
    formula: "Minimize J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²\n\nμₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ",
    formulaNote: "J = within-cluster sum of squares, μₖ = centroid of cluster k, Cₖ = set of points in cluster k",
    example: {
      title: "Customer Segmentation for Marketing",
      desc: "An e-commerce company clusters 100,000 customers using purchase frequency and average spend. K=4 reveals: Cluster 1 (Budget Regulars): frequent purchases, low spend. Cluster 2 (Whales): frequent, high spend. Cluster 3 (Window Shoppers): rare visits, low spend. Cluster 4 (Splurgers): rare but huge purchases. Each segment gets a tailored marketing strategy."
    },
    code: `from sklearn.cluster import KMeans
import numpy as np

# Customer data: [purchase_frequency, avg_spend]
X = np.array([[30,25], [35,30], [5,100], [8,120],
              [40,20], [2,15], [3,10], [7,90],
              [32,28], [6,110], [1,8], [38,22]])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

for i, center in enumerate(kmeans.cluster_centers_):
    count = (labels == i).sum()
    print(f"Cluster {i}: center=({center[0]:.0f}, \${center[1]:.0f}), "
          f"size={count}")

# Assign new customer
new_customer = np.array([[20, 50]])
print(f"New customer → Cluster {kmeans.predict(new_customer)[0]}")`,
    pros: ["Simple, fast, and scalable to large datasets", "Easy to interpret cluster centers", "Guaranteed to converge", "Works well with spherical, evenly-sized clusters"],
    cons: ["Must choose K in advance", "Sensitive to initialization (use k-means++)", "Assumes spherical clusters of equal size", "Fails on non-convex or overlapping clusters", "Sensitive to outliers"],
    complexity: { training: "O(n · K · d · iterations)", prediction: "O(K · d)", space: "O(n · d + K · d)" }
  },

  "dbscan": {
    name: "DBSCAN",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "Density-Based Spatial Clustering — finds clusters of any shape by grouping dense regions, and automatically detects outliers.",
    steps: [
      "For each point, count how many neighbors are within radius ε (epsilon).",
      "If a point has ≥ minPts neighbors within ε, it's a core point — the heart of a cluster.",
      "If a point is within ε of a core point but has < minPts neighbors, it's a border point.",
      "Points that aren't near any core point are labeled as noise (outliers).",
      "Connect all core points that are within ε of each other into the same cluster. Border points join the cluster of their nearest core point."
    ],
    formula: "Nε(p) = {q ∈ D | dist(p,q) ≤ ε}\n\nCore point: |Nε(p)| ≥ minPts\nBorder point: |Nε(p)| < minPts, but within ε of a core point\nNoise: neither core nor border",
    formulaNote: "ε (epsilon) = neighborhood radius, minPts = minimum neighbors to form a dense region",
    example: {
      title: "Detecting Earthquake Clusters",
      desc: "Seismologists plot earthquake epicenters on a map. K-Means would fail because earthquake clusters follow fault lines (arbitrary shapes, not spheres). DBSCAN with ε=50km and minPts=5 discovers: elongated clusters along the San Andreas Fault, circular clusters near volcanic hotspots, and isolated quakes marked as noise. No need to guess the number of clusters."
    },
    code: `from sklearn.cluster import DBSCAN
import numpy as np

# Geographic coordinates of events
X = np.array([[1,2],[1.5,1.8],[1.2,2.1],[5,8],[5.2,7.8],
              [5.1,8.2],[8,2],[8.1,2.2],[50,50]])  # last = outlier

db = DBSCAN(eps=1.0, min_samples=3)
labels = db.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
for i in range(n_clusters):
    points = X[labels == i]
    print(f"Cluster {i}: {len(points)} points, "
          f"center=({points.mean(0)[0]:.1f}, {points.mean(0)[1]:.1f})")`,
    pros: ["No need to specify number of clusters", "Finds clusters of any shape", "Robust to outliers (labels them as noise)", "Works well with spatial data"],
    cons: ["Sensitive to ε and minPts parameters", "Struggles with clusters of varying density", "Not deterministic for border points", "Slow on large datasets without spatial indexing"],
    complexity: { training: "O(n · log n) with R-tree, O(n²) without", prediction: "N/A (transductive)", space: "O(n)" }
  },

  "pca": {
    name: "PCA (Principal Component Analysis)",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "Reduces dimensionality by finding the directions (principal components) that capture the most variance in the data.",
    steps: [
      "Standardize the data (zero mean, unit variance) so all features are on equal footing.",
      "Compute the covariance matrix of the features.",
      "Calculate eigenvectors and eigenvalues of the covariance matrix. Eigenvectors are the principal component directions; eigenvalues indicate how much variance each captures.",
      "Sort eigenvectors by eigenvalue (descending). The first PC captures the most variance.",
      "Project data onto the top-k eigenvectors to reduce from d dimensions to k dimensions."
    ],
    formula: "C = (1/n) XᵀX    (covariance matrix)\n\nCv = λv    (eigenvalue decomposition)\n\nX_reduced = X · V_k    (project onto top-k eigenvectors)",
    formulaNote: "λ = eigenvalue (variance explained), v = eigenvector (direction), V_k = matrix of top-k eigenvectors",
    example: {
      title: "Visualizing Handwritten Digits",
      desc: "The MNIST dataset has 28×28 pixel images = 784 dimensions. PCA reduces this to 2 dimensions for visualization. The first PC captures stroke thickness, the second captures slant angle. When plotted, digits naturally cluster: 0s and 1s separate clearly, while 4s and 9s overlap (they look similar). 95% of the variance is captured by just 150 components — an 80% reduction in dimensions."
    },
    code: `from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np

# Load 8x8 digit images (64 features)
digits = load_digits()
X, y = digits.data, digits.target

# Reduce 64 dimensions → 2 for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_2d.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%}")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%}")

# How many components for 95% variance?
pca_full = PCA(n_components=0.95)
X_95 = pca_full.fit_transform(X)
print(f"Components for 95% variance: {pca_full.n_components_}")`,
    pros: ["Dramatically reduces computation for downstream models", "Removes noise in low-variance dimensions", "Enables visualization of high-dimensional data", "Removes multicollinearity between features"],
    cons: ["Components are linear combinations — hard to interpret", "Only captures linear relationships", "Information loss is inevitable", "Sensitive to feature scaling"],
    complexity: { training: "O(d²·n + d³)", prediction: "O(n·d·k)", space: "O(d·k)" }
  },

  "t-sne": {
    name: "t-SNE",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "A non-linear dimensionality reduction technique designed for visualizing high-dimensional data in 2D/3D.",
    steps: [
      "Compute pairwise similarities in the high-dimensional space using Gaussian distributions.",
      "Define a similar probability distribution in the low-dimensional space using Student's t-distribution (heavier tails).",
      "Minimize the KL divergence between the two distributions using gradient descent.",
      "The t-distribution in low-D prevents the 'crowding problem' — far-apart points stay far apart.",
      "The result is a 2D/3D embedding where local structure (nearby neighbors) is preserved."
    ],
    formula: "High-D similarity: pⱼ|ᵢ = exp(-||xᵢ-xⱼ||²/2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ-xₖ||²/2σᵢ²)\n\nLow-D similarity: qᵢⱼ = (1+||yᵢ-yⱼ||²)⁻¹ / Σₖ≠ₗ(1+||yₖ-yₗ||²)⁻¹\n\nMinimize KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)",
    formulaNote: "Perplexity parameter (5-50) controls the effective number of neighbors considered",
    example: {
      title: "Visualizing Word Embeddings",
      desc: "Word2Vec produces 300-dimensional vectors for each word. t-SNE maps them to 2D, revealing semantic clusters: country names group together, verbs cluster separately, and 'king - man + woman ≈ queen' relationships become visible as geometric patterns. What was invisible in 300D becomes a clear, beautiful map of language."
    },
    code: `from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# Reduce 64D → 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

print(f"Shape: {X.shape} → {X_embedded.shape}")
# Each digit class should form a visible cluster
for digit in range(10):
    mask = y == digit
    center = X_embedded[mask].mean(axis=0)
    print(f"Digit {digit}: center=({center[0]:.1f}, {center[1]:.1f})")`,
    pros: ["Excellent at preserving local neighborhood structure", "Reveals clusters invisible in high dimensions", "Beautiful visualizations for exploratory analysis", "Non-linear — captures complex manifold structure"],
    cons: ["Only for visualization (2D/3D), not general reduction", "Non-deterministic — results vary between runs", "Very slow on large datasets", "Cannot embed new points (must re-run on full data)", "Perplexity tuning is important"],
    complexity: { training: "O(n² · d) or O(n · log n) with Barnes-Hut", prediction: "N/A (transductive)", space: "O(n²) or O(n)" }
  },

  "autoencoders": {
    name: "Autoencoders",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "Neural networks that learn compressed representations by encoding input into a bottleneck layer, then decoding it back.",
    steps: [
      "The encoder network compresses input x into a low-dimensional latent representation z (the bottleneck).",
      "The decoder network reconstructs the input from z, producing x̂.",
      "Train by minimizing reconstruction loss: ||x - x̂||² — the output should match the input.",
      "The bottleneck forces the network to learn the most important features.",
      "After training, the encoder alone can be used for dimensionality reduction or feature extraction."
    ],
    formula: "Encoder: z = f(Wx + b)\nDecoder: x̂ = g(W'z + b')\n\nLoss = ||x - x̂||²    (reconstruction error)\n\nVariational: Loss = Reconstruction + KL(q(z|x) || p(z))",
    formulaNote: "z = latent code, the bottleneck dimension controls compression ratio",
    example: {
      title: "Anomaly Detection in Manufacturing",
      desc: "A factory trains an autoencoder on sensor data from 1,000 normal machine operations (vibration, temperature, pressure). The autoencoder learns to reconstruct 'normal' patterns with low error. When a bearing starts to fail, the vibration pattern changes — the autoencoder's reconstruction error spikes from 0.02 to 3.5. Any error above threshold 0.5 triggers a maintenance alert."
    },
    code: `import numpy as np

# Simple autoencoder concept (using numpy for clarity)
# In practice, use PyTorch or TensorFlow

# Simulated normal sensor data (100 samples, 10 features)
np.random.seed(42)
X_normal = np.random.randn(100, 10) * 0.5 + 2.0

# Anomalous data has different patterns
X_anomaly = np.random.randn(5, 10) * 2.0 + 5.0

# Concept: after training, measure reconstruction error
# Normal data: low error (model learned this pattern)
# Anomalous data: high error (model never saw this)

# Simulated reconstruction errors
normal_errors = np.random.exponential(0.1, 100)
anomaly_errors = np.random.exponential(2.0, 5)

threshold = 0.5
print(f"Normal avg error: {normal_errors.mean():.3f}")
print(f"Anomaly avg error: {anomaly_errors.mean():.3f}")
print(f"Anomalies detected: {(anomaly_errors > threshold).sum()}/5")`,
    pros: ["Learn non-linear dimensionality reduction", "Powerful for anomaly detection", "Variational autoencoders can generate new data", "Flexible architecture design", "Pre-training for other tasks"],
    cons: ["Harder to train than PCA", "No closed-form solution — requires gradient descent", "May learn trivial identity mapping", "Reconstruction quality depends on architecture choices"],
    complexity: { training: "O(epochs · n · params)", prediction: "O(n · params)", space: "O(params)" }
  },

  "apriori": {
    name: "Apriori",
    category: "Unsupervised Learning",
    badge: "unsupervised",
    subtitle: "Discovers frequent itemsets and association rules in transactional data — the algorithm behind 'customers who bought X also bought Y'.",
    steps: [
      "Scan all transactions to find items that appear above a minimum support threshold.",
      "Generate candidate pairs from frequent single items and count their occurrences.",
      "Prune pairs below the minimum support threshold.",
      "Repeat: combine frequent k-itemsets to form (k+1)-itemsets, count, and prune.",
      "From frequent itemsets, generate association rules and filter by confidence and lift."
    ],
    formula: "Support(A) = count(A) / total_transactions\nConfidence(A→B) = Support(A∪B) / Support(A)\nLift(A→B) = Confidence(A→B) / Support(B)",
    formulaNote: "High support = frequent, high confidence = reliable rule, lift > 1 = positive association",
    example: {
      title: "Supermarket Basket Analysis",
      desc: "From 10,000 receipts: {bread, butter} appears in 3,000 (support=30%). Of transactions with bread, 60% also have butter (confidence=60%). Butter alone appears in 40% of transactions, so lift = 0.60/0.40 = 1.5 — buying bread makes you 1.5x more likely to buy butter. The store places them nearby and offers bundle deals."
    },
    code: `# Using mlxtend library for Apriori
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Transaction data (1=bought, 0=not)
data = {'bread':  [1,1,0,1,1,0,1,1],
        'butter': [1,1,0,1,0,0,1,0],
        'milk':   [0,1,1,1,1,1,0,1],
        'eggs':   [0,0,1,1,0,1,0,1]}
df = pd.DataFrame(data)

# Find frequent itemsets (min 30% support)
frequent = apriori(df, min_support=0.3, use_colnames=True)

# Generate rules (min 50% confidence)
rules = association_rules(frequent, metric="confidence",
                         min_threshold=0.5)
for _, r in rules.iterrows():
    print(f"{set(r['antecedents'])} → {set(r['consequents'])}  "
          f"conf={r['confidence']:.0%} lift={r['lift']:.2f}")`,
    pros: ["Intuitive and easy to explain to stakeholders", "Finds surprising, actionable patterns", "Well-suited for transactional/basket data", "Controllable with support/confidence thresholds"],
    cons: ["Exponential candidate generation with many items", "Slow on large datasets with low support thresholds", "Generates many redundant rules", "Only finds co-occurrence, not causation"],
    complexity: { training: "O(2^d) worst case, pruning helps", prediction: "O(rules)", space: "O(frequent itemsets)" }
  },

  "q-learning": {
    name: "Q-Learning",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "A model-free RL algorithm that learns the value of taking each action in each state through trial and error.",
    steps: [
      "Initialize a Q-table with zeros: Q(state, action) for every state-action pair.",
      "Agent observes current state s and chooses action a (ε-greedy: usually best action, sometimes random).",
      "Execute action a, observe reward r and new state s'.",
      "Update Q-value: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)].",
      "Repeat millions of times. The Q-table converges to optimal values — the agent learns the best action for every state."
    ],
    formula: "Q(s,a) ← Q(s,a) + α[r + γ · maxₐ' Q(s',a') - Q(s,a)]\n\nPolicy: π(s) = argmaxₐ Q(s,a)",
    formulaNote: "α = learning rate, γ = discount factor (0-1), ε = exploration rate",
    example: {
      title: "Robot Learning to Navigate a Maze",
      desc: "A robot in a 5×5 grid maze learns to reach the exit. State = grid position (25 states), actions = up/down/left/right. Reward: -1 per step (encourages speed), +100 for reaching exit, -50 for hitting walls. After 10,000 episodes, the Q-table reveals the shortest path: the robot has learned to take optimal turns without ever seeing the maze layout — purely from trial and error."
    },
    code: `import numpy as np

# Simple gridworld Q-learning
states = 16  # 4x4 grid
actions = 4  # up, down, left, right
Q = np.zeros((states, actions))
alpha = 0.1   # learning rate
gamma = 0.95  # discount factor
epsilon = 0.1 # exploration rate
goal = 15     # bottom-right corner

for episode in range(5000):
    state = 0  # start top-left
    for step in range(50):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[state])

        # Simple transition (simplified)
        next_state = min(state + [-4, 4, -1, 1][action], states-1)
        next_state = max(next_state, 0)
        reward = 100 if next_state == goal else -1

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state
        if state == goal:
            break

print("Learned Q-values (state 0):", Q[0].round(1))`,
    pros: ["Model-free — doesn't need environment dynamics", "Guaranteed to converge to optimal policy (with conditions)", "Simple to implement", "Off-policy — can learn from other agents' experiences"],
    cons: ["Q-table explodes with large/continuous state spaces", "Slow convergence in complex environments", "No generalization between similar states", "Requires careful tuning of α, γ, ε"],
    complexity: { training: "O(episodes · steps)", prediction: "O(actions)", space: "O(states · actions)" }
  },

  "dqn": {
    name: "Deep Q-Network (DQN)",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "Replaces Q-learning's table with a deep neural network — enabling RL in environments with massive state spaces like video games.",
    steps: [
      "Replace the Q-table with a neural network: Q(s,a; θ) takes state s as input, outputs Q-values for all actions.",
      "Agent interacts with environment, stores (s, a, r, s') transitions in a replay buffer.",
      "Sample random mini-batches from the buffer (experience replay) to break correlation between consecutive samples.",
      "Train the network to minimize: [r + γ·max Q(s',a'; θ⁻) - Q(s,a; θ)]² using a target network θ⁻ for stability.",
      "Periodically copy main network weights to the target network."
    ],
    formula: "Loss = 𝔼[(r + γ · maxₐ' Q(s',a'; θ⁻) - Q(s,a; θ))²]\n\nTwo key innovations:\n1. Experience Replay: store and randomly sample transitions\n2. Target Network: separate network θ⁻ updated periodically",
    formulaNote: "θ = main network weights, θ⁻ = target network weights (updated every C steps)",
    example: {
      title: "Mastering Atari Breakout from Pixels",
      desc: "DeepMind's DQN takes 4 stacked 84×84 grayscale frames as input (to capture motion). The CNN processes raw pixels → Q-values for 4 actions (left, right, fire, noop). After 10 million frames of play, the agent discovers the tunnel strategy: break through one side and bounce the ball behind the wall — a strategy the developers didn't program. It achieves superhuman scores on 29 of 49 Atari games."
    },
    code: `# DQN pseudocode (conceptual — full impl needs PyTorch)
import numpy as np
from collections import deque

class DQN:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=10000)  # replay buffer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # In practice: build CNN/MLP with PyTorch
        # self.model = build_network(state_size, action_size)
        # self.target_model = build_network(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # explore
        # return argmax(self.model.predict(state))       # exploit

    def replay(self, batch_size=32):
        # Sample random batch from memory
        # Compute target: r + γ * max(target_model(s'))
        # Train model on (state, target) pairs
        # Decay epsilon
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)`,
    pros: ["Handles high-dimensional state spaces (images, sensors)", "Experience replay improves sample efficiency", "Target network stabilizes training", "Generalized to many Atari games with same architecture"],
    cons: ["Overestimates Q-values (mitigated by Double DQN)", "Only works with discrete action spaces", "Requires millions of frames to train", "Sensitive to hyperparameters and architecture"],
    complexity: { training: "O(episodes · steps · network_forward_pass)", prediction: "O(forward_pass)", space: "O(network_params + replay_buffer)" }
  },

  "policy-gradient": {
    name: "Policy Gradient (REINFORCE)",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "Directly optimizes the policy (action probabilities) by following the gradient of expected reward — no Q-values needed.",
    steps: [
      "Parameterize the policy as a neural network: π(a|s; θ) outputs action probabilities given state.",
      "Run a complete episode, collecting states, actions, and rewards.",
      "Compute the return (cumulative discounted reward) for each timestep.",
      "Compute the policy gradient: ∇J = 𝔼[∇log π(a|s) · G_t] — increase probability of good actions.",
      "Update policy parameters: θ ← θ + α · ∇J. Repeat for many episodes."
    ],
    formula: "∇J(θ) = 𝔼[Σₜ ∇log π(aₜ|sₜ; θ) · Gₜ]\n\nGₜ = Σₖ₌₀ γᵏ rₜ₊ₖ    (return from time t)\n\nWith baseline: ∇J(θ) = 𝔼[∇log π(aₜ|sₜ; θ) · (Gₜ - b)]",
    formulaNote: "Actions with high returns get their probability increased, low returns get decreased. Baseline b reduces variance.",
    example: {
      title: "Training a Game-Playing Agent",
      desc: "For a simple game like CartPole: the policy network takes the pole angle and cart position as input, outputs probabilities for left/right. After 500 episodes, the agent learns: when the pole tilts right, push right (high probability). The key insight: unlike Q-learning, policy gradient naturally handles continuous or stochastic policies."
    },
    code: `# REINFORCE pseudocode
import numpy as np

class PolicyGradientAgent:
    def __init__(self, lr=0.01, gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        # self.policy_net = build_network()

    def compute_returns(self):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        # Normalize returns (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        returns = self.compute_returns()
        # loss = -Σ log_prob * return  (negative for gradient ascent)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

# Training loop:
# for episode in range(1000):
#     state = env.reset()
#     while not done:
#         action = agent.select_action(state)
#         state, reward, done = env.step(action)
#         agent.rewards.append(reward)
#     agent.update()`,
    pros: ["Works with continuous action spaces", "Can learn stochastic policies", "Directly optimizes what we care about (expected reward)", "Simpler than value-based methods for some problems"],
    cons: ["High variance — needs many episodes to converge", "Sample inefficient (on-policy)", "Can converge to local optima", "Credit assignment is difficult for long episodes"],
    complexity: { training: "O(episodes · steps · network)", prediction: "O(forward_pass)", space: "O(network_params)" }
  },

  "ppo": {
    name: "PPO (Proximal Policy Optimization)",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "The industry-standard policy optimization algorithm — stable, reliable, and used to train everything from robots to ChatGPT (via RLHF).",
    steps: [
      "Collect a batch of trajectories using the current policy.",
      "Compute advantages: how much better each action was compared to average (using GAE).",
      "Compute the probability ratio: r(θ) = π_new(a|s) / π_old(a|s).",
      "Clip the ratio to [1-ε, 1+ε] to prevent too-large policy updates.",
      "Optimize the clipped surrogate objective for multiple epochs on the same batch."
    ],
    formula: "L(θ) = 𝔼[min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)]\n\nrₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)\n\nε = 0.2 (clipping range)",
    formulaNote: "The clipping prevents destructively large policy updates, making training stable",
    example: {
      title: "RLHF for Language Models",
      desc: "To make an LLM helpful and safe: (1) Generate responses with the current model. (2) A reward model scores each response based on human preferences. (3) PPO updates the LLM to increase the probability of high-scoring responses while clipping prevents it from changing too drastically. This is how ChatGPT and Claude are fine-tuned — PPO keeps the model stable while steering it toward human-preferred behavior."
    },
    code: `# PPO update pseudocode
import numpy as np

def ppo_update(policy, old_log_probs, states, actions,
               returns, advantages, clip_eps=0.2, epochs=4):
    for epoch in range(epochs):
        # Get current policy probabilities
        # new_log_probs = policy.log_prob(states, actions)

        # Probability ratio
        # ratio = exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        # surr1 = ratio * advantages
        # surr2 = clip(ratio, 1-clip_eps, 1+clip_eps) * advantages
        # loss = -min(surr1, surr2).mean()

        # Also add value loss and entropy bonus
        # total_loss = loss + 0.5*value_loss - 0.01*entropy
        pass

# PPO training loop:
# while not converged:
#     trajectories = collect_rollouts(policy, env, n_steps=2048)
#     advantages = compute_gae(trajectories, value_fn)
#     for epoch in range(4):
#         for batch in mini_batches(trajectories):
#             ppo_update(policy, batch)`,
    pros: ["Very stable training — the clipping prevents catastrophic updates", "Sample efficient (reuses data for multiple epochs)", "Works for both discrete and continuous actions", "Industry standard — used at OpenAI, DeepMind, Anthropic", "Simple to implement relative to other advanced RL"],
    cons: ["Still sample-inefficient compared to off-policy methods", "Hyperparameter sensitive (clip range, learning rate)", "Requires good advantage estimation (GAE)", "Can converge to suboptimal policies"],
    complexity: { training: "O(batch_size · epochs · network)", prediction: "O(forward_pass)", space: "O(network_params + trajectory_buffer)" }
  },

  "actor-critic": {
    name: "Actor-Critic",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "Combines the best of policy gradient (actor) and value estimation (critic) for lower-variance, more stable RL training.",
    steps: [
      "Actor network: π(a|s; θ) — selects actions based on the current policy.",
      "Critic network: V(s; w) — estimates how good the current state is (value function).",
      "Agent takes action, receives reward. Critic computes TD error: δ = r + γV(s') - V(s).",
      "Update critic to minimize TD error: w ← w + α_c · δ · ∇V(s).",
      "Update actor using critic's feedback: θ ← θ + α_a · δ · ∇log π(a|s). The critic's value estimate replaces noisy returns."
    ],
    formula: "TD error (advantage): δₜ = rₜ + γV(sₜ₊₁; w) - V(sₜ; w)\n\nCritic update: w ← w + αc · δₜ · ∇wV(sₜ; w)\nActor update: θ ← θ + αa · δₜ · ∇θ log π(aₜ|sₜ; θ)",
    formulaNote: "The critic reduces variance of the policy gradient by providing a learned baseline",
    example: {
      title: "Self-Driving Car Steering",
      desc: "The actor takes camera images and outputs steering angle (continuous). The critic evaluates 'how well am I doing?' based on lane centering, smoothness, and safety. When the car veers left, the critic gives a low value → the actor learns to steer right. The critic's instant feedback (TD learning) is much faster than waiting for episode-end returns."
    },
    code: `# A2C (Advantage Actor-Critic) pseudocode
class ActorCritic:
    def __init__(self):
        # self.actor = PolicyNetwork()   # outputs action probs
        # self.critic = ValueNetwork()   # outputs state value
        self.gamma = 0.99

    def update(self, state, action, reward, next_state, done):
        # Critic: estimate values
        # value = self.critic(state)
        # next_value = 0 if done else self.critic(next_state)

        # TD error = advantage estimate
        # td_error = reward + self.gamma * next_value - value

        # Update critic (minimize TD error²)
        # critic_loss = td_error ** 2

        # Update actor (policy gradient with advantage)
        # log_prob = log(self.actor(state)[action])
        # actor_loss = -log_prob * td_error.detach()

        # total_loss = actor_loss + 0.5 * critic_loss
        # total_loss.backward()
        # optimizer.step()
        pass

# Variants: A2C (synchronous), A3C (async parallel),
# SAC (entropy-regularized), TD3 (twin delayed)`,
    pros: ["Lower variance than pure policy gradient", "Online learning (updates every step, not every episode)", "Foundation for advanced methods (A3C, SAC, PPO)", "Works with continuous action spaces"],
    cons: ["Two networks to train — more complex", "Critic bias can mislead the actor", "Hyperparameter sensitive (two learning rates)", "Can be unstable if critic is inaccurate"],
    complexity: { training: "O(steps · (actor_pass + critic_pass))", prediction: "O(actor_forward_pass)", space: "O(actor_params + critic_params)" }
  },

  "alphago": {
    name: "AlphaGo / AlphaZero",
    category: "Reinforcement Learning",
    badge: "reinforcement",
    subtitle: "Combined deep neural networks with Monte Carlo Tree Search to master Go — then generalized to learn any board game from scratch.",
    steps: [
      "Neural network takes board state as input, outputs: (a) policy — probability of each move, (b) value — probability of winning.",
      "Monte Carlo Tree Search (MCTS) uses the network to guide search: explore promising moves more deeply.",
      "During MCTS, each simulation traverses the tree, evaluates leaf positions with the network, and backpropagates results.",
      "After MCTS, play the most-visited move. Store (board_state, MCTS_policy, game_outcome) as training data.",
      "AlphaZero: train purely from self-play. No human games needed. The network improves → MCTS improves → generates better training data → feedback loop."
    ],
    formula: "MCTS selection: a* = argmax[Q(s,a) + c · P(s,a) · √N(s) / (1+N(s,a))]\n\nNetwork: (p, v) = fθ(s)\n  p = move probabilities, v = win probability\n\nLoss = (z-v)² - πᵀlog p + c||θ||²",
    formulaNote: "Q = action value, P = prior from network, N = visit count, z = actual game outcome, π = MCTS policy",
    example: {
      title: "Mastering Go from Zero Knowledge",
      desc: "AlphaGo Zero starts with random play and zero human knowledge. After 3 days of self-play (4.9 million games), it defeated the original AlphaGo (which trained on thousands of human games) 100-0. After 40 days, it surpassed all previous versions. It discovered novel Go strategies that surprised professional players — moves that humans had never considered in 3,000 years of play."
    },
    code: `# AlphaZero self-play conceptual pseudocode
class AlphaZero:
    def __init__(self):
        # self.network = DualHeadNetwork()  # policy + value heads
        self.mcts_simulations = 800

    def self_play_game(self):
        training_data = []
        # state = initial_board()
        # while not game_over(state):
        #     # Run MCTS guided by neural network
        #     mcts_policy = mcts_search(state, self.network,
        #                                self.mcts_simulations)
        #     training_data.append((state, mcts_policy))
        #     action = sample(mcts_policy)  # with temperature
        #     state = apply_move(state, action)
        # outcome = get_winner(state)
        # return [(s, p, outcome) for s, p in training_data]
        pass

    def train(self, n_iterations=1000):
        for i in range(n_iterations):
            # Generate self-play games
            # games = [self.self_play_game() for _ in range(100)]
            # Train network on collected data
            # network.train(games)  # minimize policy + value loss
            pass

# AlphaZero mastered Chess, Go, and Shogi
# with the SAME algorithm — only the game rules changed`,
    pros: ["Achieved superhuman performance in Go, Chess, and Shogi", "Learns from scratch — no human expertise needed", "Self-play creates infinite training data", "General framework applicable to any perfect-information game"],
    cons: ["Requires enormous compute (thousands of TPUs/GPUs)", "Only works for perfect-information, two-player games", "MCTS is slow at inference time", "Not directly applicable to real-world continuous control"],
    complexity: { training: "O(games · moves · MCTS_sims · network)", prediction: "O(MCTS_sims · network)", space: "O(network_params + tree_size)" }
  },

  "cnn": {
    name: "CNN (Convolutional Neural Network)",
    category: "Deep Learning",
    badge: "deep-learning",
    subtitle: "Neural networks that use learnable filters to automatically extract spatial features from grid-like data such as images.",
    steps: [
      "Input: an image (e.g., 224×224×3 for RGB). Each pixel is a feature.",
      "Convolutional layers: small filters (e.g., 3×3) slide across the image, computing dot products. Each filter detects a specific pattern (edges, corners, textures).",
      "Activation (ReLU): introduces non-linearity — sets negative values to zero.",
      "Pooling layers: downsample feature maps (e.g., 2×2 max pooling) to reduce spatial dimensions and gain translation invariance.",
      "Deeper layers combine low-level features into high-level concepts: edges → textures → parts → objects.",
      "Fully connected layers at the end map features to output classes."
    ],
    formula: "Convolution: (f * g)(i,j) = ΣₘΣₙ f(m,n) · g(i-m, j-n)\n\nOutput size: (W - F + 2P) / S + 1\n\nReLU: f(x) = max(0, x)",
    formulaNote: "W = input size, F = filter size, P = padding, S = stride. A 3×3 filter on a 32×32 image → 30×30 output (no padding)",
    example: {
      title: "Detecting Pneumonia from Chest X-Rays",
      desc: "A CNN takes a 224×224 chest X-ray as input. Layer 1 filters detect edges and bright spots. Layer 2 combines edges into rib outlines and lung boundaries. Layer 3 detects opacity patterns. Layer 4 recognizes consolidation patterns typical of pneumonia. The final layer outputs: 92% pneumonia, 8% normal. Grad-CAM heatmaps highlight exactly which lung region triggered the diagnosis."
    },
    code: `import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224→224
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 224→112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 112→112
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 112→56
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 56→56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # 56→1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(num_classes=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: ~200K parameters for this simple model
# ResNet-50 has ~25M, EfficientNet-B7 has ~66M`,
    pros: ["Automatic feature extraction — no manual engineering", "Translation invariant (detects cats anywhere in image)", "Parameter sharing (same filter across entire image) = efficient", "State-of-the-art for image tasks", "Transfer learning: pre-trained models work for new tasks"],
    cons: ["Requires large labeled datasets", "Computationally expensive (GPU needed)", "Not equivariant to rotation or scale by default", "Pooling loses spatial precision", "Can be fooled by adversarial examples"],
    complexity: { training: "O(epochs · n · layers · K² · Cin · Cout · H · W)", prediction: "O(layers · K² · Cin · Cout · H · W)", space: "O(parameters + feature maps)" }
  },

  "rnn-lstm": {
    name: "RNN / LSTM",
    category: "Deep Learning",
    badge: "deep-learning",
    subtitle: "Recurrent networks that process sequences step-by-step, maintaining a hidden state that acts as memory of past inputs.",
    steps: [
      "At each time step t, the RNN takes input xₜ and the previous hidden state hₜ₋₁.",
      "Compute new hidden state: hₜ = tanh(Wₓₓxₜ + Wₕₕhₜ₋₁ + b). This 'remembers' past information.",
      "Problem: vanilla RNNs suffer from vanishing gradients — they forget long-range dependencies.",
      "LSTM solves this with a cell state (long-term memory) controlled by three gates: Forget gate (what to discard), Input gate (what to store), Output gate (what to output).",
      "The cell state acts as a conveyor belt — information flows through unchanged unless a gate modifies it."
    ],
    formula: "RNN: hₜ = tanh(Wₓₓxₜ + Wₕₕhₜ₋₁ + b)\n\nLSTM gates:\n  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)     — forget gate\n  iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)      — input gate\n  cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ] + bc)  — cell state\n  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)      — output gate\n  hₜ = oₜ ⊙ tanh(cₜ)               — hidden state",
    formulaNote: "σ = sigmoid (0-1, acts as a gate), ⊙ = element-wise multiplication",
    example: {
      title: "Predicting Next Word in a Sentence",
      desc: "Input: 'The cat sat on the ___'. The LSTM processes each word sequentially. After 'The', hidden state encodes 'article seen'. After 'cat', it encodes 'subject is animal'. After 'sat on the', the forget gate clears irrelevant info and the cell state encodes 'animal sitting on surface'. The output layer predicts 'mat' (45%), 'floor' (20%), 'chair' (15%) — favoring words that complete the spatial context."
    },
    code: `import torch
import torch.nn as nn

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,
                 hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                           num_layers=num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)        # (batch, seq, embed)
        output, (h, c) = self.lstm(embedded) # output at each step
        # Use last timestep for prediction
        logits = self.fc(output[:, -1, :])
        return logits

model = TextLSTM(vocab_size=10000)
# Input: sequence of word indices
x = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10000) = next word probs`,
    pros: ["Handles variable-length sequences naturally", "LSTM solves the vanishing gradient problem", "Cell state preserves long-range dependencies", "Bidirectional variants capture context from both directions", "Well-suited for time series and sequential data"],
    cons: ["Sequential processing — cannot parallelize (slow training)", "Still struggles with very long sequences (>500 tokens)", "Complex architecture with many parameters per cell", "Largely replaced by Transformers for text/NLP tasks"],
    complexity: { training: "O(seq_len · hidden² · epochs · n)", prediction: "O(seq_len · hidden²)", space: "O(hidden² · layers)" }
  },

  "transformer": {
    name: "Transformer",
    category: "Deep Learning",
    badge: "deep-learning",
    subtitle: "The architecture behind GPT, Claude, BERT, and most modern AI — uses self-attention to process entire sequences in parallel.",
    steps: [
      "Tokenize input into tokens. Add positional encodings so the model knows word order.",
      "Self-Attention: each token computes Query, Key, and Value vectors. Attention score = softmax(QKᵀ/√d).",
      "Each token attends to ALL other tokens simultaneously — capturing relationships regardless of distance.",
      "Multi-Head Attention: run multiple attention heads in parallel, each learning different relationship types.",
      "Feed-Forward Network: each position independently passes through a two-layer MLP.",
      "Stack N layers (GPT-3 has 96 layers). Each layer refines the representations."
    ],
    formula: "Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V\n\nMultiHead = Concat(head₁,...,headₕ)Wᴼ\nwhere headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)\n\nPositional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d))",
    formulaNote: "dₖ = key dimension, h = number of heads (typically 8-96), √dₖ scaling prevents softmax saturation",
    example: {
      title: "How Claude Understands Your Question",
      desc: "When you ask 'What is the capital of France?', the Transformer tokenizes this into ~8 tokens. In self-attention, 'capital' strongly attends to 'France' (high attention score), connecting the concept. 'What' attends to 'is' (question structure). Each of 80+ layers refines understanding: early layers handle syntax, middle layers handle semantics, later layers handle reasoning. The final layer generates 'Paris' token by token."
    },
    code: `import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        # Compute Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1,2).contiguous().view(B, T, C)
        return self.W_o(out)

attn = SelfAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq=10, dim=512
print(f"Output: {attn(x).shape}")  # (2, 10, 512)`,
    pros: ["Parallelizable — processes all tokens simultaneously (fast training)", "Captures long-range dependencies without degradation", "Scales extremely well (GPT-4 has >1T parameters)", "Transfer learning: pre-train once, fine-tune for any task", "Unified architecture for text, images, audio, code"],
    cons: ["O(n²) attention — quadratic in sequence length", "Requires enormous compute and data to train from scratch", "No inherent notion of position (needs positional encoding)", "Difficult to interpret what attention heads learn", "Energy-intensive"],
    complexity: { training: "O(n² · d · layers · epochs)", prediction: "O(n² · d · layers) per token", space: "O(parameters + n² · heads)" }
  },

  "gan": {
    name: "GAN (Generative Adversarial Network)",
    category: "Deep Learning",
    badge: "deep-learning",
    subtitle: "Two neural networks compete — a Generator creates fake data while a Discriminator tries to catch it. The competition produces stunningly realistic outputs.",
    steps: [
      "Generator G takes random noise z and produces a fake sample: x_fake = G(z).",
      "Discriminator D takes a sample (real or fake) and outputs probability it's real.",
      "Train D to correctly classify real vs fake: maximize log(D(x_real)) + log(1-D(G(z))).",
      "Train G to fool D: minimize log(1-D(G(z))) — make fakes that D thinks are real.",
      "Alternate training. Over time, G produces increasingly realistic outputs and D becomes better at detecting fakes — until equilibrium."
    ],
    formula: "min_G max_D V(D,G) = 𝔼[log D(x)] + 𝔼[log(1 - D(G(z)))]\n\nGenerator: G: z → x_fake  (noise → data)\nDiscriminator: D: x → [0,1]  (data → real probability)\n\nNash equilibrium: D(x) = 0.5 for all x",
    formulaNote: "At equilibrium, the discriminator can't tell real from fake — outputs 50/50 for everything",
    example: {
      title: "Generating Photorealistic Faces (StyleGAN)",
      desc: "StyleGAN2 generates 1024×1024 face images of people who don't exist. The generator takes 512-dim noise and maps it through a style network that controls: coarse features (face shape, pose) in early layers, medium features (eyes, nose, mouth) in middle layers, and fine details (skin texture, hair strands) in late layers. Each generated face is unique, photorealistic, and has never existed — yet is indistinguishable from a real photo."
    },
    code: `import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh(),  # 28x28 image
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()
z = torch.randn(16, 100)  # 16 random noise vectors
fake_images = G(z)
predictions = D(fake_images)
print(f"Generated: {fake_images.shape}")     # (16,1,28,28)
print(f"D scores: {predictions[:4].squeeze().tolist()}")`,
    pros: ["Generates remarkably realistic images, audio, and video", "Learns the full data distribution implicitly", "No explicit density estimation needed", "Creative applications: style transfer, super-resolution, inpainting"],
    cons: ["Training is notoriously unstable (mode collapse, vanishing gradients)", "Hard to evaluate quality objectively", "No convergence guarantees", "Generator can produce biased or harmful content", "Largely supplanted by diffusion models for image generation"],
    complexity: { training: "O(epochs · n · (G_params + D_params))", prediction: "O(G_forward_pass)", space: "O(G_params + D_params)" }
  },

  "bfs": {
    name: "Breadth-First Search (BFS)",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Explores a graph level by level — guaranteed to find the shortest path in unweighted graphs.",
    steps: [
      "Start at the source node. Add it to a queue.",
      "Dequeue the front node. Mark it as visited.",
      "Explore all unvisited neighbors and add them to the queue.",
      "Repeat until the goal is found or the queue is empty.",
      "The first time a node is reached is guaranteed to be via the shortest path (in unweighted graphs)."
    ],
    formula: "Uses a FIFO Queue\n\nTime: O(V + E)   — visits every vertex and edge once\nSpace: O(V)      — stores all vertices at current level\n\nGuarantees: Complete (always finds solution if one exists)\n            Optimal (shortest path in unweighted graphs)",
    formulaNote: "V = vertices, E = edges. BFS explores in concentric 'rings' outward from the source.",
    example: {
      title: "Finding Shortest Route in a Social Network",
      desc: "LinkedIn's 'degrees of connection': to find the shortest path from you to Elon Musk, BFS starts with your direct connections (level 1), then their connections (level 2), and so on. At level 3, it finds a path: You → Alice → Bob → Elon. BFS guarantees this is the minimum number of hops — no shorter connection exists."
    },
    code: `from collections import deque

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path  # shortest path found!

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # no path exists

# Social network graph
graph = {
    'You': ['Alice', 'Carol'],
    'Alice': ['You', 'Bob', 'Dave'],
    'Bob': ['Alice', 'Elon'],
    'Carol': ['You', 'Dave'],
    'Dave': ['Alice', 'Carol'],
    'Elon': ['Bob'],
}

path = bfs(graph, 'You', 'Elon')
print(f"Shortest path: {' → '.join(path)}")
print(f"Degrees of separation: {len(path) - 1}")`,
    pros: ["Guaranteed shortest path in unweighted graphs", "Complete — will find a solution if one exists", "Simple to implement with a queue", "Naturally finds all nodes at distance k"],
    cons: ["High memory usage — stores entire frontier", "Slow for deep solutions (explores all shallow nodes first)", "Not suitable for weighted graphs (use Dijkstra/A*)", "Exponential space in branching factor"],
    complexity: { training: "N/A", prediction: "O(V + E)", space: "O(V)" }
  },

  "dfs": {
    name: "Depth-First Search (DFS)",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Explores as deep as possible before backtracking — memory efficient but doesn't guarantee shortest paths.",
    steps: [
      "Start at the source node. Push it onto a stack (or use recursion).",
      "Pop the top node. Mark it as visited.",
      "Push all unvisited neighbors onto the stack.",
      "Repeat — DFS always goes deeper before exploring siblings.",
      "When a dead end is reached, backtrack to the most recent node with unexplored neighbors."
    ],
    formula: "Uses a LIFO Stack (or recursion)\n\nTime: O(V + E)\nSpace: O(V) worst case, O(bm) typical — only stores current path\n\nComplete: Only in finite graphs\nOptimal: No — may find a long path before a short one",
    formulaNote: "b = branching factor, m = maximum depth. DFS uses much less memory than BFS.",
    example: {
      title: "Solving a Maze",
      desc: "In a maze, DFS picks a direction and follows it as far as possible. Hit a wall? Backtrack to the last intersection and try a different path. This is exactly how you'd solve a maze by hand — keep your left hand on the wall. DFS might find a long winding path instead of the shortest one, but it uses very little memory (only the current path)."
    },
    code: `def dfs(graph, start, goal, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == goal:
        return path[:]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result = dfs(graph, neighbor, goal, visited, path)
            if result:
                return result

    path.pop()  # backtrack
    return None

# Maze as adjacency list
maze = {
    'Start': ['A', 'B'],
    'A': ['Start', 'C', 'D'],
    'B': ['Start', 'E'],
    'C': ['A'],
    'D': ['A', 'Exit'],
    'E': ['B', 'F'],
    'F': ['E'],
    'Exit': ['D'],
}

path = dfs(maze, 'Start', 'Exit')
print(f"Path found: {' → '.join(path)}")
print(f"Path length: {len(path) - 1} steps")`,
    pros: ["Very memory efficient — only stores current path", "Simple recursive implementation", "Good for detecting cycles and topological sorting", "Finds any path quickly in deep graphs"],
    cons: ["Does NOT find shortest path", "Can get stuck in infinite loops without cycle detection", "May explore very deep paths unnecessarily", "Not complete in infinite graphs"],
    complexity: { training: "N/A", prediction: "O(V + E)", space: "O(b·m) — depth × branching" }
  },

  "a-star": {
    name: "A* Search",
    category: "Classical AI",
    badge: "classical",
    subtitle: "The gold standard pathfinding algorithm — combines actual cost with a heuristic estimate to find optimal paths efficiently.",
    steps: [
      "Maintain a priority queue ordered by f(n) = g(n) + h(n).",
      "g(n) = actual cost from start to node n. h(n) = heuristic estimate from n to goal.",
      "Expand the node with lowest f(n) — the most promising path.",
      "For each neighbor: if the new path is cheaper, update its g-value and re-queue.",
      "When the goal is dequeued, the path is guaranteed optimal (if h never overestimates)."
    ],
    formula: "f(n) = g(n) + h(n)\n\ng(n) = actual cost from start to n\nh(n) = estimated cost from n to goal (heuristic)\n\nAdmissible: h(n) ≤ actual cost (never overestimates)\nConsistent: h(n) ≤ cost(n,n') + h(n')",
    formulaNote: "Common heuristics: Euclidean distance, Manhattan distance. Admissible h guarantees optimality.",
    example: {
      title: "GPS Navigation",
      desc: "Finding the fastest route from San Francisco to Los Angeles. g(n) = actual driving time so far. h(n) = straight-line distance / max_speed (admissible — you can't drive faster than the limit). A* expands inland cities only if they're on potentially faster routes, ignoring clearly bad options. It finds the optimal route via I-5 (5h 52m) without exploring the scenic Pacific Coast Highway (8h) because h(n) reveals it can't be faster."
    },
    code: `import heapq

def a_star(graph, start, goal, h):
    open_set = [(h(start), 0, start, [start])]  # (f, g, node, path)
    visited = {}

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, g  # optimal path and cost

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        for neighbor, cost in graph[current]:
            new_g = g + cost
            new_f = new_g + h(neighbor)
            heapq.heappush(open_set, (new_f, new_g, neighbor,
                                       path + [neighbor]))

    return None, float('inf')

# Road network: node → [(neighbor, distance)]
roads = {
    'SF':  [('SJ', 50), ('Sac', 90)],
    'SJ':  [('SF', 50), ('Fresno', 150)],
    'Sac': [('SF', 90), ('Fresno', 170)],
    'Fresno': [('SJ', 150), ('Sac', 170), ('LA', 220)],
    'LA':  [('Fresno', 220)],
}

# Heuristic: straight-line distance to LA
h_to_la = {'SF': 350, 'SJ': 300, 'Sac': 380, 'Fresno': 220, 'LA': 0}

path, cost = a_star(roads, 'SF', 'LA', lambda n: h_to_la[n])
print(f"Optimal route: {' → '.join(path)}")
print(f"Total distance: {cost} miles")`,
    pros: ["Guaranteed optimal path with admissible heuristic", "Much more efficient than BFS/Dijkstra (heuristic guides search)", "Complete (will find a solution if one exists)", "Used in virtually every pathfinding application"],
    cons: ["Memory-intensive — stores all explored nodes", "Performance depends heavily on heuristic quality", "Not suitable for dynamic environments (must re-run)", "Exponential worst case with poor heuristic"],
    complexity: { training: "N/A", prediction: "O(b^d) worst, much better with good h", space: "O(b^d)" }
  },

  "greedy-best-first": {
    name: "Greedy Best-First Search",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Expands the node that appears closest to the goal — fast but not optimal, as it ignores the cost already traveled.",
    steps: [
      "Maintain a priority queue ordered by h(n) only (heuristic estimate to goal).",
      "Always expand the node that appears closest to the goal.",
      "Unlike A*, ignores g(n) — the cost already paid to reach the current node.",
      "Very fast when the heuristic is good, but can be misled by obstacles.",
      "Neither complete (can loop) nor optimal (ignores path cost)."
    ],
    formula: "f(n) = h(n)    (only heuristic, no actual cost!)\n\nCompare to A*: f(n) = g(n) + h(n)\n\nGreedy ignores g(n), so it may take expensive paths\nthat appear to head toward the goal.",
    formulaNote: "Fast but can find suboptimal or even no solution. Use A* when optimality matters.",
    example: {
      title: "Quick-and-Dirty Navigation",
      desc: "Imagine driving toward a mountain. Greedy always heads straight toward it (minimizing straight-line distance). But there's a lake in the way — greedy goes right to the shore and gets stuck. A* would have gone around because it considers the actual cost. Greedy works great in open terrain but fails with obstacles."
    },
    code: `import heapq

def greedy_best_first(graph, start, goal, h):
    open_set = [(h(start), start, [start])]
    visited = set()

    while open_set:
        _, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                heapq.heappush(open_set,
                    (h(neighbor), neighbor, path + [neighbor]))

    return None

# Same road network
roads = {
    'A': [('B',1), ('C',10)],
    'B': [('A',1), ('D',5)],
    'C': [('A',10), ('D',1)],
    'D': [('B',5), ('C',1)],
}
h = {'A': 8, 'B': 6, 'C': 2, 'D': 0}

path = greedy_best_first(roads, 'A', 'D', lambda n: h[n])
print(f"Greedy path: {' → '.join(path)}")
# May find A→C→D (cost=11) instead of optimal A→B→D (cost=6)`,
    pros: ["Very fast — minimal node expansion", "Simple to implement", "Good when heuristic is very accurate", "Uses less memory than A*"],
    cons: ["NOT optimal — may find expensive paths", "NOT complete — can loop or miss solutions", "Easily misled by obstacles", "Performance entirely depends on heuristic quality"],
    complexity: { training: "N/A", prediction: "O(b^m) worst", space: "O(b^m)" }
  },

  "genetic": {
    name: "Genetic Algorithm",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Optimization inspired by evolution — maintains a population of solutions that improve through selection, crossover, and mutation.",
    steps: [
      "Initialize a random population of candidate solutions (chromosomes).",
      "Evaluate fitness of each individual using the objective function.",
      "Selection: choose parents with probability proportional to fitness (tournament, roulette wheel).",
      "Crossover: combine two parents to create offspring (single-point, uniform, or custom crossover).",
      "Mutation: randomly alter some genes with small probability to maintain diversity.",
      "Replace the old population with offspring. Repeat for many generations."
    ],
    formula: "Selection probability: P(i) = fitness(i) / Σ fitness(j)\n\nCrossover (single-point):\n  Parent1: [A B C|D E F]    Child1: [A B C|d e f]\n  Parent2: [a b c|d e f] →  Child2: [a b c|D E F]\n\nMutation: flip random gene with probability p_m ≈ 0.01",
    formulaNote: "Population size ~100-1000, mutation rate ~1%, crossover rate ~70-90%",
    example: {
      title: "Optimizing Delivery Routes (TSP)",
      desc: "A delivery company with 50 stops needs the shortest route. There are 50! ≈ 3×10⁶⁴ possible routes — brute force is impossible. GA approach: each chromosome is a route permutation. Start with 200 random routes. The fittest (shortest) routes are more likely to be parents. Crossover combines good sub-routes from two parents. Mutation swaps two random stops. After 500 generations, the route is within 5% of optimal — found in seconds."
    },
    code: `import random
import numpy as np

def genetic_algorithm(fitness_fn, gene_length, pop_size=100,
                      generations=200, mutation_rate=0.01):
    # Initialize random population
    population = [np.random.randint(0, 2, gene_length)
                  for _ in range(pop_size)]

    for gen in range(generations):
        # Evaluate fitness
        scores = [fitness_fn(ind) for ind in population]
        best_idx = np.argmax(scores)
        best = scores[best_idx]

        if gen % 50 == 0:
            print(f"Gen {gen}: best fitness = {best:.2f}")

        # Selection (tournament)
        new_pop = []
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            parent1 = population[i if scores[i]>scores[j] else j]
            i, j = random.sample(range(pop_size), 2)
            parent2 = population[i if scores[i]>scores[j] else j]

            # Crossover
            point = random.randint(1, gene_length - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])

            # Mutation
            for k in range(gene_length):
                if random.random() < mutation_rate:
                    child[k] = 1 - child[k]
            new_pop.append(child)

        population = new_pop

    scores = [fitness_fn(ind) for ind in population]
    return population[np.argmax(scores)]

# Example: maximize number of 1s in a binary string
result = genetic_algorithm(lambda x: x.sum(), gene_length=50)
print(f"Best solution: {result.sum()}/50 ones")`,
    pros: ["Works with any objective function (no gradient needed)", "Explores broadly — good at escaping local optima", "Handles discrete, continuous, and mixed search spaces", "Naturally parallelizable", "Intuitive and flexible"],
    cons: ["No guarantee of finding the global optimum", "Slow convergence compared to gradient-based methods", "Many hyperparameters (population size, rates, selection method)", "Fitness function evaluations can be expensive"],
    complexity: { training: "O(generations · pop_size · fitness_eval)", prediction: "O(1) — use best solution", space: "O(pop_size · gene_length)" }
  },

  "simulated-annealing": {
    name: "Simulated Annealing",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Optimization inspired by metallurgy — accepts worse solutions with decreasing probability, allowing escape from local optima.",
    steps: [
      "Start with a random solution and a high temperature T.",
      "Generate a random neighbor (small modification of current solution).",
      "If the neighbor is better, always accept it.",
      "If the neighbor is worse, accept it with probability P = e^(-ΔE/T) — high temperature = more likely to accept.",
      "Gradually reduce T (cooling schedule). Early on, bad moves are accepted freely (exploration). Later, only improvements are accepted (exploitation).",
      "Stop when T reaches minimum or no improvement for many iterations."
    ],
    formula: "Accept probability: P(accept) = e^(-ΔE / T)\n\nΔE = f(neighbor) - f(current)\n\nCooling: T(t) = T₀ · α^t   (geometric, α ≈ 0.995)\n  or:    T(t) = T₀ / (1 + β·t)  (linear)",
    formulaNote: "High T → accept almost anything. Low T → accept only improvements. ΔE > 0 means worse.",
    example: {
      title: "Optimizing Chip Layout Design",
      desc: "A semiconductor company needs to place 1,000 components on a chip to minimize wire length. Start with random placement. Each step: randomly move one component. If total wire length decreases → accept. If it increases by ΔE, accept with probability e^(-ΔE/T). At T=1000, wild rearrangements are accepted (escaping poor layouts). At T=0.01, only tiny improvements pass. Final result: near-optimal placement that no greedy algorithm could find."
    },
    code: `import math
import random

def simulated_annealing(cost_fn, initial, neighbor_fn,
                        T_start=1000, T_min=0.001, alpha=0.995):
    current = initial
    current_cost = cost_fn(current)
    best = current[:]
    best_cost = current_cost
    T = T_start

    while T > T_min:
        neighbor = neighbor_fn(current)
        neighbor_cost = cost_fn(neighbor)
        delta = neighbor_cost - current_cost

        # Accept better solutions always;
        # accept worse solutions with probability e^(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        T *= alpha  # cool down

    return best, best_cost

# Example: minimize f(x) = x² (find x closest to 0)
def cost(x):
    return x[0]**2

def neighbor(x):
    return [x[0] + random.gauss(0, 1)]

best, cost_val = simulated_annealing(cost, [10.0], neighbor)
print(f"Best x = {best[0]:.4f}, f(x) = {cost_val:.6f}")`,
    pros: ["Simple to implement", "Can escape local optima (unlike hill climbing)", "Works with any objective function", "Single solution — low memory", "Good for combinatorial optimization"],
    cons: ["Cooling schedule is hard to tune", "No guarantee of finding global optimum", "Slow convergence in large search spaces", "Only one solution explored at a time", "Results vary between runs"],
    complexity: { training: "O(iterations · neighbor_eval)", prediction: "O(1)", space: "O(solution_size)" }
  },

  "hill-climbing": {
    name: "Hill Climbing",
    category: "Classical AI",
    badge: "classical",
    subtitle: "The simplest optimization — always move to the best neighboring state. Fast but easily trapped in local optima.",
    steps: [
      "Start with an initial solution (random or heuristic).",
      "Evaluate all neighboring solutions (small modifications).",
      "Move to the best neighbor if it improves the objective.",
      "If no neighbor is better, stop — you're at a local optimum.",
      "Variants: steepest ascent (check all neighbors), stochastic (random uphill neighbor), random restart (repeat with new start)."
    ],
    formula: "current ← initial_solution\nrepeat:\n  neighbor ← best(neighbors(current))\n  if eval(neighbor) > eval(current):\n    current ← neighbor\n  else:\n    return current  // local optimum\n\nRandom restart: repeat N times, keep best result",
    formulaNote: "Pure hill climbing is greedy — it only goes uphill. Random restarts partially solve the local optima problem.",
    example: {
      title: "Tuning Neural Network Hyperparameters",
      desc: "Start with learning_rate=0.01, batch_size=32. Check neighbors: lr=0.005 (accuracy 85%), lr=0.02 (87%), batch=64 (86%). Move to lr=0.02 (best). Check again: lr=0.015 (86%), lr=0.025 (86.5%). No improvement → stuck at local optimum. Random restart hill climbing: try 10 random starting points, hill climb from each, keep the best. Often finds 89% accuracy."
    },
    code: `import random

def hill_climbing(cost_fn, initial, neighbor_fn, max_iters=1000):
    current = initial
    current_cost = cost_fn(current)

    for i in range(max_iters):
        neighbor = neighbor_fn(current)
        neighbor_cost = cost_fn(neighbor)

        if neighbor_cost < current_cost:  # minimizing
            current = neighbor
            current_cost = neighbor_cost

    return current, current_cost

def random_restart_hill_climbing(cost_fn, random_fn, neighbor_fn,
                                  restarts=10):
    best = None
    best_cost = float('inf')

    for _ in range(restarts):
        solution, cost = hill_climbing(cost_fn, random_fn(),
                                        neighbor_fn)
        if cost < best_cost:
            best = solution
            best_cost = cost

    return best, best_cost

# Minimize f(x,y) = x² + y² (with random restarts)
result, cost = random_restart_hill_climbing(
    cost_fn=lambda p: p[0]**2 + p[1]**2,
    random_fn=lambda: [random.uniform(-10,10), random.uniform(-10,10)],
    neighbor_fn=lambda p: [p[0]+random.gauss(0,0.5),
                           p[1]+random.gauss(0,0.5)],
    restarts=20
)
print(f"Best: ({result[0]:.3f}, {result[1]:.3f}), cost={cost:.6f}")`,
    pros: ["Extremely simple to implement", "Very fast for simple landscapes", "Low memory usage", "Good starting point before trying complex optimizers", "Random restart variant is surprisingly effective"],
    cons: ["Trapped by local optima (biggest limitation)", "Plateaus cause stagnation", "Ridges are hard to navigate", "No guarantee of finding global optimum", "Performance depends heavily on starting point"],
    complexity: { training: "O(iterations · neighbors)", prediction: "O(1)", space: "O(solution_size)" }
  },

  "bayesian-networks": {
    name: "Bayesian Networks",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Directed graphical models that represent probabilistic cause-and-effect relationships — enabling principled reasoning under uncertainty.",
    steps: [
      "Define the structure: a directed acyclic graph (DAG) where nodes are variables and edges represent causal influence.",
      "For each node, specify a conditional probability table (CPT): P(node | parents).",
      "The joint distribution factorizes: P(X₁,...,Xₙ) = Π P(Xᵢ | Parents(Xᵢ)).",
      "Inference: given observed evidence, compute posterior probabilities of other variables using Bayes' theorem.",
      "Can be learned from data (structure learning) or built from expert knowledge."
    ],
    formula: "Bayes' Theorem: P(A|B) = P(B|A)·P(A) / P(B)\n\nChain rule: P(X₁,...,Xₙ) = Π P(Xᵢ | Parents(Xᵢ))\n\nNaive Bayes: P(y|x₁,...,xₙ) ∝ P(y)·Π P(xᵢ|y)",
    formulaNote: "The DAG encodes conditional independencies — nodes are independent of non-descendants given parents.",
    example: {
      title: "Medical Diagnosis System",
      desc: "A diagnostic network: Smoking → Cancer, Smoking → Bronchitis, Cancer → X-ray positive, Cancer → Shortness of breath, Bronchitis → Shortness of breath. A patient has shortness of breath but a normal X-ray. Without the network: P(Cancer) is moderate. With the network: the normal X-ray strongly reduces P(Cancer), but P(Bronchitis) increases to explain the shortness of breath. The network reasons: 'Bronchitis is more likely because it explains the symptom without contradicting the X-ray.'"
    },
    code: `# Using pgmpy library for Bayesian Networks
# pip install pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define structure
model = BayesianNetwork([
    ('Smoking', 'Cancer'),
    ('Smoking', 'Bronchitis'),
    ('Cancer', 'Xray'),
    ('Cancer', 'Dyspnea'),
    ('Bronchitis', 'Dyspnea'),
])

# Define conditional probability tables
cpd_smoking = TabularCPD('Smoking', 2, [[0.7], [0.3]])
cpd_cancer = TabularCPD('Cancer', 2, [[0.95,0.85],[0.05,0.15]],
                        evidence=['Smoking'], evidence_card=[2])
cpd_bronch = TabularCPD('Bronchitis', 2, [[0.6,0.3],[0.4,0.7]],
                        evidence=['Smoking'], evidence_card=[2])
cpd_xray = TabularCPD('Xray', 2, [[0.8,0.1],[0.2,0.9]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dysp = TabularCPD('Dyspnea', 2,
    [[0.9,0.3,0.4,0.1],[0.1,0.7,0.6,0.9]],
    evidence=['Cancer','Bronchitis'], evidence_card=[2,2])

model.add_cpds(cpd_smoking,cpd_cancer,cpd_bronch,cpd_xray,cpd_dysp)

# Query: P(Cancer | Dyspnea=yes, Xray=normal)
infer = VariableElimination(model)
result = infer.query(['Cancer'], evidence={'Dyspnea':1, 'Xray':0})
print(result)`,
    pros: ["Interpretable causal reasoning", "Handles missing data naturally", "Combines expert knowledge with data-driven learning", "Principled uncertainty quantification", "Efficient inference via conditional independence"],
    cons: ["Structure learning is NP-hard", "Requires domain expertise to build correctly", "Exact inference is intractable for large networks", "Assumes no cycles (DAG only)", "Discrete variables require discretization"],
    complexity: { training: "O(n · 2^max_parents) for parameter learning", prediction: "O(2^treewidth) for exact inference", space: "O(n · 2^max_parents)" }
  },

  "hmm": {
    name: "Hidden Markov Model",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Models sequences where the true state is hidden — you only observe noisy emissions. Fundamental to speech recognition and bioinformatics.",
    steps: [
      "Define hidden states (e.g., weather: Sunny/Rainy) and observable emissions (e.g., activities: Walk/Shop/Clean).",
      "Specify transition probabilities: P(stateₜ | stateₜ₋₁) — how states evolve over time.",
      "Specify emission probabilities: P(observationₜ | stateₜ) — what each state 'emits'.",
      "Given a sequence of observations, use the Viterbi algorithm to find the most likely state sequence.",
      "Use Forward-Backward algorithm for P(state at time t | all observations). Use Baum-Welch (EM) to learn parameters from data."
    ],
    formula: "Transition: A[i,j] = P(sₜ=j | sₜ₋₁=i)\nEmission: B[j,k] = P(oₜ=k | sₜ=j)\nInitial: π[i] = P(s₁=i)\n\nViterbi: δₜ(j) = max_i [δₜ₋₁(i) · A[i,j]] · B[j,oₜ]",
    formulaNote: "The Markov property: future depends only on present state, not on how we got there.",
    example: {
      title: "Speech Recognition (Phoneme Detection)",
      desc: "Hidden states = phonemes (/k/, /æ/, /t/ for 'cat'). Observations = audio frequency features. The HMM models: P(next phoneme | current phoneme) as transition matrix, and P(audio features | phoneme) as emission. Given an audio recording, Viterbi finds the most likely phoneme sequence. This was the backbone of speech recognition before deep learning (Siri, Dragon Dictation)."
    },
    code: `import numpy as np

def viterbi(observations, states, start_p, trans_p, emit_p):
    """Find most likely hidden state sequence."""
    n_states = len(states)
    T = len(observations)

    # dp[t][s] = probability of most likely path ending in state s at time t
    dp = np.zeros((T, n_states))
    backtrack = np.zeros((T, n_states), dtype=int)

    # Initialize
    for s in range(n_states):
        dp[0][s] = start_p[s] * emit_p[s][observations[0]]

    # Fill
    for t in range(1, T):
        for s in range(n_states):
            probs = dp[t-1] * trans_p[:, s] * emit_p[s][observations[t]]
            dp[t][s] = np.max(probs)
            backtrack[t][s] = np.argmax(probs)

    # Backtrack
    path = [np.argmax(dp[-1])]
    for t in range(T-1, 0, -1):
        path.insert(0, backtrack[t][path[0]])

    return [states[s] for s in path]

# Weather example: hidden=Sunny/Rainy, observed=Walk/Shop/Clean
states = ['Sunny', 'Rainy']
start_p = np.array([0.6, 0.4])
trans_p = np.array([[0.7, 0.3], [0.4, 0.6]])  # Sunny→Sunny=0.7
emit_p = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])  # Sunny→Walk=0.6

observations = [0, 1, 2, 0]  # Walk, Shop, Clean, Walk
path = viterbi(observations, states, start_p, trans_p, emit_p)
print(f"Observations: Walk, Shop, Clean, Walk")
print(f"Hidden states: {' → '.join(path)}")`,
    pros: ["Elegant mathematical framework for sequences", "Efficient exact inference (Viterbi is O(T·S²))", "Handles noisy observations naturally", "Well-understood theory with convergence guarantees"],
    cons: ["Markov assumption is often too restrictive", "Limited to discrete states (need extensions for continuous)", "Cannot model long-range dependencies", "Replaced by RNNs/Transformers for most NLP tasks", "Number of states must be fixed"],
    complexity: { training: "O(T · S² · iterations) for Baum-Welch", prediction: "O(T · S²) for Viterbi", space: "O(T · S)" }
  },

  "monte-carlo": {
    name: "Monte Carlo Methods",
    category: "Classical AI",
    badge: "classical",
    subtitle: "Use random sampling to estimate complex quantities — when you can't compute the answer exactly, simulate it millions of times.",
    steps: [
      "Define what you want to estimate (an integral, a probability, an expected value).",
      "Design a random sampling procedure that generates relevant samples.",
      "Run many simulations (thousands to millions), collecting results.",
      "The sample average converges to the true value by the Law of Large Numbers.",
      "More samples = more accuracy. Error decreases as O(1/√n)."
    ],
    formula: "Estimate: E[f(X)] ≈ (1/N) Σ f(xᵢ)  where xᵢ ~ P(x)\n\nError ∝ 1/√N  (to halve error, need 4× more samples)\n\nπ estimation: throw random darts at square,\n  π ≈ 4 × (darts in circle / total darts)",
    formulaNote: "Monte Carlo works for ANY problem you can simulate — even if you can't solve it analytically.",
    example: {
      title: "Estimating Pi by Random Sampling",
      desc: "Draw a unit square with an inscribed quarter circle (radius=1). Randomly throw 1,000,000 darts. Count how many land inside the quarter circle (x² + y² ≤ 1). Ratio = 785,421 / 1,000,000 = 0.785. Since the quarter circle area = π/4, we get π ≈ 4 × 0.785 = 3.1417 — very close to 3.14159! This is Monte Carlo: solving geometry with randomness."
    },
    code: `import numpy as np

# 1. Estimate Pi using random darts
def estimate_pi(n_samples=1000000):
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * inside.mean()

pi_est = estimate_pi()
print(f"Estimated Pi: {pi_est:.5f} (actual: 3.14159)")

# 2. Monte Carlo Integration: ∫₀¹ x² dx = 1/3
def mc_integrate(f, a, b, n=100000):
    x = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(x))

result = mc_integrate(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx ≈ {result:.5f} (actual: 0.33333)")

# 3. Risk simulation: portfolio returns
returns = np.random.normal(0.08, 0.15, (10000, 252))  # daily
final = 10000 * np.prod(1 + returns/252, axis=1)
print(f"Portfolio after 1 year:")
print(f"  Mean: \${final.mean():,.0f}")
print(f"  5th percentile (VaR): \${np.percentile(final, 5):,.0f}")
print(f"  P(loss): {(final < 10000).mean():.1%}")`,
    pros: ["Works for any problem you can simulate", "Simple to implement and parallelize", "Handles high-dimensional integrals easily", "No assumptions about distribution shape", "Accuracy improves predictably with more samples"],
    cons: ["Slow convergence: O(1/√N)", "Can be computationally expensive for high accuracy", "Variance can be high without variance reduction techniques", "Random seed affects reproducibility", "Not suitable for deterministic problems with exact solutions"],
    complexity: { training: "N/A", prediction: "O(N · simulation_cost)", space: "O(N) for storing samples" }
  }
};

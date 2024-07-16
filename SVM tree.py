class SVMTreeNode:
    def __init__(self, depth=0, max_depth=3):
        self.depth = depth
        self.max_depth = max_depth
        self.model = None
        self.left = None
        self.right = None
        self.feature_indices = None

    def fit(self, X, y):
        if self.depth < self.max_depth and len(y) > 1:
            # 두 개의 특징을 선택
            self.feature_indices = np.random.choice(X.shape[1], 2, replace=False)
            X_selected = X[:, self.feature_indices]
            
            # SVM 훈련
            self.model = SVC(kernel='linear')
            self.model.fit(X_selected, y)
            
            # 예측 및 데이터 분할
            predictions = self.model.predict(X_selected)
            left_indices = np.where(predictions == 0)[0]
            right_indices = np.where(predictions == 1)[0]
            
            # 자식 노드 생성
            if len(left_indices) > 0:
                self.left = SVMTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
                self.left.fit(X[left_indices], y[left_indices])
            if len(right_indices) > 0:
                self.right = SVMTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
                self.right.fit(X[right_indices], y[right_indices])

    def predict(self, X):
        if self.model is None:
            return np.zeros(X.shape[0])
        
        X_selected = X[:, self.feature_indices]
        predictions = self.model.predict(X_selected)
        
        left_indices = np.where(predictions == 0)[0]
        right_indices = np.where(predictions == 1)[0]
        
        y_pred = np.zeros(X.shape[0])
        if self.left is not None and len(left_indices) > 0:
            y_pred[left_indices] = self.left.predict(X[left_indices])
        if self.right is not None and len(right_indices) > 0:
            y_pred[right_indices] = self.right.predict(X[right_indices])
        
        return y_pred

# SVM-Tree 모델 훈련
svm_tree = SVMTreeNode(max_depth=3)
svm_tree.fit(X_train, y_train)

# 예측
y_pred_train = svm_tree.predict(X_train)
y_pred_test = svm_tree.predict(X_test)

# 정확도 평가
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

train_accuracy, test_accuracy

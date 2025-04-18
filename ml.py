import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

real_data_path = "adult.csv"  
synthetic_data_path = "cleaned_data.csv"  

real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(synthetic_data_path)

def preprocess_data(df, target_column):
 
    df = df.dropna()

    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


target_column = "income"
X_real, y_real = preprocess_data(real_df, target_column)
X_synthetic, y_synthetic = preprocess_data(synthetic_df, target_column)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(X_synthetic, y_synthetic, test_size=0.2,
                                                                    random_state=42)


def evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, y_pred_rf)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = accuracy_score(y_test, y_pred_knn)

    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)

    gbm = GradientBoostingClassifier()
    gbm.fit(X_train, y_train)
    y_pred_gbm = gbm.predict(X_test)
    results['GBM'] = accuracy_score(y_test, y_pred_gbm)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    y_pred_ann = model.predict(X_test)
    y_pred_ann = (y_pred_ann > 0.5).astype(int)  
    results['ANN'] = accuracy_score(y_test, y_pred_ann)

    return results

print("Evaluating on Real Dataset...")
results_real = evaluate_models(X_train_real, X_test_real, y_train_real, y_test_real)

print("\nEvaluating on Synthetic Dataset...")
results_syn = evaluate_models(X_train_syn, X_test_syn, y_train_syn, y_test_syn)

print("\nPerformance on Real Dataset:")
for model, score in results_real.items():
    print(f"{model}: {score:.4f}")

print("\nPerformance on Synthetic Dataset:")
for model, score in results_syn.items():
    print(f"{model}: {score:.4f}")

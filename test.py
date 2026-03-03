
########################################################################
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 모델 관련
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# 언더샘플링
from imblearn.under_sampling import RandomUnderSampler

# 부스터 모델
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 딥러닝 (CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# -------------------------------
# 0. 데이터 불러오기
X = pd.read_csv("X_mm.csv")
y = pd.read_csv("y.csv").squeeze()

# y 값 확인 및 변환 - 모든 클래스를 0, 1로 매핑
print("원본 y 값:", y.unique())
unique_vals = sorted(y.unique())

if len(unique_vals) == 2:
    # 이미 2개 클래스인 경우 0, 1로 변환
    y = y.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(int)
elif len(unique_vals) == 3:
    # 3개 클래스인 경우 처리 방법 선택
    print("⚠️ 3개의 클래스가 발견되었습니다:", unique_vals)
    print("클래스별 샘플 수:")
    print(y.value_counts().sort_index())

    # 옵션 1: 가장 적은 클래스를 제거하고 이진 분류로 변환
    # 옵션 2: 다중 클래스 분류로 전환 (코드 전체 수정 필요)
    # 여기서는 옵션 1을 선택

    # 클래스 0과 2만 사용 (클래스 1 제거)
    mask = y != 1
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # 0, 2를 0, 1로 매핑
    y = y.map({0: 0, 2: 1}).astype(int)
    print("변환 후 y 값:", y.unique())
else:
    raise ValueError(f"예상치 못한 클래스 개수: {len(unique_vals)}")

print("✓ 데이터 로드 완료:", X.shape, y.shape, "Unique y:", y.unique())
print("클래스 분포:", y.value_counts().to_dict())


# -------------------------------
# 1. 평가 함수
def evaluate_model(name, y_true, y_pred, y_proba, fold):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = (recall + specificity) / 2
    auc = roc_auc_score(y_true, y_proba)
    return {
        "Fold": fold,
        "Algorithm": name,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "AUC": auc
    }


# -------------------------------
# 2. 파라미터 후보 (최소화)
param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Random Forest": {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    "XGBoost": {'n_estimators': [100], 'max_depth': [3, 5]},
    "LightGBM": {'n_estimators': [100], 'num_leaves': [31, 50]},
    "CatBoost": {'depth': [4, 6], 'iterations': [100], 'learning_rate': [0.1]},
    "MLP": {'hidden_layer_sizes': [(64,), (64, 32)], 'alpha': [0.0001, 0.001]}
}

# -------------------------------
# 3. 기본 모델
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "MLP": MLPClassifier(max_iter=300, random_state=42)
}

# -------------------------------
# 4. 학습 + 평가 (5-Fold, 메모리 최적화)
results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 스케일링
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_valid_scaled = scaler.transform(X_valid).astype(np.float32)

    # ------------------ 🛠️ 이 부분에 추가 및 수정 🛠️ ------------------
    # 언더샘플링 (원래 코드에서 오류 발생)

    # 1. y_train의 클래스별 샘플 수 확인
    target_counts = y_train.value_counts()

    # 2. 다수 클래스 수 & 소수 클래스 확인
    major_class_count = target_counts.max()
    minority_class = target_counts.index[target_counts.argmin()]

    # 3. 목표 소수 클래스 샘플 수 계산 (다수 클래스의 50%)
    # NOTE: RandomUnderSampler는 '다수 클래스'를 줄여 목표 샘플 수에 맞춥니다.
    # float(0.5)를 썼다는 것은, '다수 클래스'를 줄여 '소수 클래스' 수의 2배가 되도록 하거나,
    # '소수 클래스'를 줄여 '다수 클래스' 수의 50%가 되도록 한다는 의미가 될 수 있습니다.
    # float 방식은 이진 분류에서 소수 클래스를 기준으로 다수 클래스의 샘플 수를 계산하여 줄입니다.
    # 안전하게 원본 로직(소수 클래스 수 = 다수 클래스 수 * 0.5)을 따르기 위해 딕셔너리를 사용합니다.
    target_minority_count = int(major_class_count * 0.5)

    # 4. 딕셔너리 형태의 sampling_strategy 생성 (소수 클래스의 목표 샘플 수 명시)
    # 다른 클래스(다수 클래스)의 샘플 수는 변화시키지 않거나,
    # imblearn이 내부적으로 알아서 다수 클래스를 줄이도록 'auto'를 사용할 수 있지만,
    # 여기서는 float=0.5의 의도에 맞게 소수 클래스의 목표치를 명시합니다.
    sampling_strategy_dict = {
        c: target_minority_count if c == minority_class else target_counts[c]
        for c in target_counts.index
    }

    # 5. 언더샘플링 (딕셔너리 사용으로 float 오류 해결)
    # ------------------ 언더샘플링 (수정) ------------------
    target_counts = y_train.value_counts()
    print(f"Fold {fold} - 학습 데이터 클래스 분포: {target_counts.to_dict()}")

    # 다수 클래스를 소수 클래스의 2배로 맞추기 (ratio=0.5는 소수:다수 = 1:2)
    # float 방식이 더 간단하고 명확합니다
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_res, y_res = rus.fit_resample(X_train_scaled, y_train)

    print(f"  → 언더샘플링 후 클래스 분포: {pd.Series(y_res).value_counts().to_dict()}")
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------

    # 메모리 절약을 위해 float32 변환
    X_res = X_res.astype(np.float32)

    # 전통 ML 모델 (RandomizedSearchCV + n_jobs=2)
    for name, model in base_models.items():
        print(f'현재 {name}으로 학습 중')
        clf = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=2, cv=3, scoring='f1',
            n_jobs=2, random_state=42
        )
        clf.fit(X_res, y_res)
        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_valid_scaled)
        y_proba = best_model.predict_proba(X_valid_scaled)[:, 1]
        results.append(evaluate_model(name, y_valid, y_pred, y_proba, fold))

    # CNN
    cnn = Sequential([
        Dense(64, activation='relu', input_shape=(X_res.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(X_res, y_res, epochs=5, batch_size=32, verbose=0)  # epochs 줄임

    y_pred_cnn = cnn.predict(X_valid_scaled).flatten()
    y_pred_cnn_label = (y_pred_cnn >= 0.5).astype(int)
    results.append(evaluate_model("CNN", y_valid, y_pred_cnn_label, y_pred_cnn, fold))

# -------------------------------
# 5. KNN 평가 (별도)
print("\nKNN 평가 시작...")
X_scaled_all = MinMaxScaler().fit_transform(X).astype(np.float32)

# 전체 데이터 언더샘플링
rus_all = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_res_all, y_res_all = rus_all.fit_resample(X_scaled_all, y)
print(f"KNN - 언더샘플링 후 클래스 분포: {pd.Series(y_res_all).value_counts().to_dict()}")

knn = KNeighborsClassifier(n_neighbors=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_knn = cross_val_predict(knn, X_res_all, y_res_all, cv=kf, method='predict')
y_proba_knn = cross_val_predict(knn, X_res_all, y_res_all, cv=kf, method='predict_proba')[:, 1]
results.append(evaluate_model("KNN", y_res_all, y_pred_knn, y_proba_knn, fold='All'))

# -------------------------------
# 6. 결과 정리 + 저장
result_df = pd.DataFrame(results)

summary = result_df.groupby("Algorithm")[[
    "Precision", "Recall", "Specificity", "F1-score",
    "Accuracy", "Balanced Accuracy", "AUC"
]].mean().round(4).sort_values("AUC", ascending=False)

result_df.to_csv("all_fold_results.csv", index=False, encoding="utf-8-sig")
summary.to_csv("model_summary.csv", encoding="utf-8-sig")

print("저장 완료")
print(" - all_fold_results.csv : Fold별 전체 성능 기록")
print(" - model_summary.csv : 모델별 평균 성능 요약")
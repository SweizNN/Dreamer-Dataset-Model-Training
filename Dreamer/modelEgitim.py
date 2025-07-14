import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#LOCO-CV (Leave-One-Subject-Out Cross Validation)

# Veri yükle
df = pd.read_csv("dreamer_eeg_features.csv")

# Özellik ve hedefler
feature_cols = [col for col in df.columns if col not in ["Valence", "Arousal", "Dominance"]]
targets = ["Valence", "Arousal", "Dominance"]
n_subjects = len(df) // 18
samples_per_subject = 18

# Modeller
model_defs = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', C=1.0, probability=True)
}

results = []
best_models = {}

for target in targets:
    print(f"\n🎯 Başlıyoruz: {target}")
    model_scores = {name: {"F1s": []} for name in model_defs}

    for test_subj in range(n_subjects):
        test_start = test_subj * samples_per_subject
        test_end = (test_subj + 1) * samples_per_subject

        test_df = df.iloc[test_start:test_end]
        train_df = df.drop(index=range(test_start, test_end), errors="ignore")


        X_train = train_df[feature_cols].values
        y_train = (train_df[target].values >= 3).astype(int)
        X_test = test_df[feature_cols].values
        y_test = (test_df[target].values >= 3).astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"⚠️ Katılımcı {test_subj+1} atlandı (etiket dengesiz)")
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in model_defs.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            results.append({
                "Target": target, "Model": name, "Subject": test_subj+1,
                "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec
            })

            model_scores[name]["F1s"].append(f1)

    # Ortalama test F1 skorları ile en iyi modeli seç
    best_model_name = max(model_scores.items(), key=lambda x: np.mean(x[1]["F1s"]))[0]
    print(f"✅ {target} için en iyi model: {best_model_name} (Ortalama Test F1 = {np.mean(model_scores[best_model_name]['F1s']):.4f})")

    # Son modeli eğit + kaydet
    X_all = df[feature_cols].values
    y_all = (df[target].values >= 3).astype(int)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)

    best_model = model_defs[best_model_name]
    best_model.fit(X_all_scaled, y_all)
    best_models[target] = {"model": best_model, "scaler": final_scaler}

# Kaydet
joblib.dump(best_models, "bestModel.pkl")
print("📦 Tüm en iyi modeller kaydedildi.")

# Sonuçları kaydet
df_results = pd.DataFrame(results)
df_results.to_csv("sonuclar.csv", index=False)

# Grafik
metrics = ["Accuracy", "F1", "Precision", "Recall"]
for metric in metrics:
    for target in targets:
        plt.figure(figsize=(10, 5))
        for name in model_defs.keys():
            data = df_results[(df_results["Target"] == target) & (df_results["Model"] == name)]
            plt.plot(data["Subject"], data[metric], marker='o', label=name)
        plt.title(f"{target} - {metric} (LOSO-CV)")
        plt.xlabel("Subject")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

#
#MATLAB
# % DREAMER EEG Özellik Çıkarımı (Z-score temizleme + log(PSD) + ReliefF)
#
# % DREAMER verisini yükle
# D = load('DREAMER.mat');
# dreamer = D.DREAMER;
#
# fs = dreamer.EEG_SamplingRate;  % 128 Hz
# n_subjects = length(dreamer.Data);
# n_videos = dreamer.noOfVideoSequences;
#
# % Bandpass filtre (4–30 Hz)
# bpFilt = designfilt('bandpassfir', 'FilterOrder', 128, ...
#     'CutoffFrequency1', 4, 'CutoffFrequency2', 30, ...
#     'SampleRate', fs);
#
# % Sonuçları sakla
# all_features = [];
# all_targets = [];
# included_subjects = [];
#
# for subj = 1:n_subjects
#     fprintf("👤 Katılımcı %d işleniyor...\n", subj);
#     participant = dreamer.Data{subj};
#     subject_features = [];
#     subject_targets = [];
#
#     for vid = 1:n_videos
#         try
#             if length(participant.EEG.stimuli) < vid || isempty(participant.EEG.stimuli{vid})
#                 fprintf("⚠️ Katılımcı %d video %d: EEG eksik\n", subj, vid);
#                 continue;
#             end
#
#             raw = participant.EEG.stimuli{vid};
#             [rows, cols] = size(raw);
#
#             if cols ~= 14 || rows < 500
#                 fprintf("⚠️ Katılımcı %d video %d: Geçersiz boyut [%d x %d]\n", subj, vid, rows, cols);
#                 continue;
#             end
#
#             raw = raw';  % [14 x zaman]
#
#             % ✅ Z-score ile kanal bazlı outlier bastırma
#             threshold = 5;
#             for ch = 1:14
#                 z = (raw(ch,:) - mean(raw(ch,:))) / std(raw(ch,:));
#                 outliers = abs(z) > threshold;
#                 raw(ch, outliers) = median(raw(ch,~outliers));  % outlier'lara medyan düzeltme
#             end
#
#             % ✅ CAR
#             mean_signal = mean(raw, 1);
#             eeg_car = raw - mean_signal;
#
#             % ✅ Bandpass
#             eeg_filt = zeros(size(eeg_car));
#             for ch = 1:14
#                 eeg_filt(ch,:) = filtfilt(bpFilt, eeg_car(ch,:));
#             end
#
#             % ✅ log(PSD)
#             window = 256;
#             noverlap = 128;
#             features = [];
#
#             for ch = 1:14
#                 [pxx, f] = pwelch(eeg_filt(ch,:), window, noverlap, [], fs);
#                 theta = bandpower(pxx, f, [4 8], 'psd');
#                 alpha = bandpower(pxx, f, [8 13], 'psd');
#                 beta  = bandpower(pxx, f, [13 20], 'psd');
#                 features = [features, log(theta + eps), log(alpha + eps), log(beta + eps)];
#             end
#
#             val = participant.ScoreValence(vid);
#             aro = participant.ScoreArousal(vid);
#             dom = participant.ScoreDominance(vid);
#             subject_features = [subject_features; features];
#             subject_targets = [subject_targets; val, aro, dom];
#
#         catch ME
#             fprintf("❌ Katılımcı %d video %d: Hata → %s\n", subj, vid, ME.message);
#         end
#     end
#
#     if size(subject_features, 1) == 18
#         all_features = [all_features; subject_features];
#         all_targets = [all_targets; subject_targets];
#         included_subjects = [included_subjects, subj];
#     else
#         fprintf("⛔ Katılımcı %d eksik video (%d/18), dışlandı\n", subj, size(subject_features,1));
#     end
# end
#
# % ================================
# % 💾 CSV olarak tüm özellikleri kaydet
# % ================================
# feature_table = array2table(all_features);
# target_table = array2table(all_targets, 'VariableNames', {'Valence','Arousal','Dominance'});
# fprintf("🧾 Dahil edilen katılımcılar: %s\n", mat2str(included_subjects));
#
#
# % ================================
# % 🔁 ReliefF: Valence, Arousal, Dominance için ayrı ayrı
# % ================================
# duygular = {'Valence', 'Arousal', 'Dominance'};
#
# for i = 1:3
#     fprintf("\n🔬 ReliefF - %s için başlatılıyor...\n", duygular{i});
#
#     % Etiketi ikili sınıfa çevir (>=3 yüksek, <3 düşük)
#     Y = double(all_targets(:, i) >= 3);
#     X = all_features;
#
#     % ReliefF ile önem sıralaması
#     [feature_ranks, feature_weights] = relieff(X, Y, 10);
#
#     % En iyi 20 özelliği seç
#     top_k = 20;
#     top_features = feature_ranks(1:top_k);
#     fprintf("📌 %s için en iyi %d özellik:\n", duygular{i}, top_k);
#     disp(top_features);
#
#     % Yeni X matrisi oluştur
#     X_selected = X(:, top_features);
#
#     % CSV olarak kaydet
#     selected_table = array2table(X_selected);
#     selected_table = [selected_table, target_table];
#     file_name = sprintf('dreamer_relief_%s.csv', lower(duygular{i}));
#     writetable(selected_table, file_name);
#
#     fprintf("✅ Kaydedildi: %s\n", file_name);
# end


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 📥 1. Verileri yükle
df_val = pd.read_csv("dreamer_relief_valence.csv")
df_aro = pd.read_csv("dreamer_relief_arousal.csv")
df_dom = pd.read_csv("dreamer_relief_dominance.csv")

#  Ortak etiket sütunlarını hizala ve kontrol et
assert df_val[["Valence", "Arousal", "Dominance"]].equals(df_aro[["Valence", "Arousal", "Dominance"]])
assert df_val[["Valence", "Arousal", "Dominance"]].equals(df_dom[["Valence", "Arousal", "Dominance"]])

#  özellikleri birleştir (60 özellik olacak: 20+20+20)
X = pd.concat([df_val.iloc[:, :-3], df_aro.iloc[:, :-3], df_dom.iloc[:, :-3]], axis=1).values
y_all = df_val[["Valence", "Arousal", "Dominance"]].values  # hedefler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', C=1.0)
}

targets = ["Valence", "Arousal", "Dominance"]
results = []

for i, target in enumerate(targets):
    print(f"\n🎯 {target} için 10-fold çapraz doğrulama başlıyor...")
    y = (y_all[:, i] >= 3).astype(int)  # ikili sınıflandırma

    for name, model in models.items():
        #10 katlı çapraz doğrulama
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring=make_scorer(f1_score))
        acc = cross_val_score(model, X_scaled, y, cv=cv, scoring=make_scorer(accuracy_score))
        prec = cross_val_score(model, X_scaled, y, cv=cv, scoring=make_scorer(precision_score))
        rec = cross_val_score(model, X_scaled, y, cv=cv, scoring=make_scorer(recall_score))

        results.append({
            "Target": target,
            "Model": name,
            "Accuracy": np.mean(acc),
            "F1": np.mean(f1),
            "Precision": np.mean(prec),
            "Recall": np.mean(rec)
        })

        print(f"✅ {name} → F1: {np.mean(f1):.4f}, Accuracy: {np.mean(acc):.4f}")

# 📊 6. Sonuçları tabloya aktar
df_results = pd.DataFrame(results)
print("\n📈 Tüm Sonuçlar:")
print(df_results)
df_results.to_csv("ReliefF_10Fold_Results.csv", index=False)
# 🎯 Hedef bazında en iyi modelleri F1 skoruna göre yazdır
print("\n🏆 Hedef Başına En İyi Modeller (F1 Skoruna Göre):")
for target in targets:
    df_target = df_results[df_results["Target"] == target]
    best_row = df_target.loc[df_target["F1"].idxmax()]
    print(f"{target}: {best_row['Model']} (F1 = {best_row['F1']:.4f})")


import pandas as pd

df = pd.read_csv("sonuçlar.csv")


overall_results = df.groupby(["Target", "Model"])[["Accuracy", "F1", "Precision", "Recall"]].mean().reset_index()
print(overall_results)


for target in ["Valence", "Arousal", "Dominance"]:
    best = df[df["Target"] == target].groupby("Model")["Accuracy"].mean().idxmax()
    acc = df[(df["Target"] == target) & (df["Model"] == best)]["Accuracy"].mean()
    f1 = df[(df["Target"] == target) & (df["Model"] == best)]["F1"].mean()
    print(f"{target} - En iyi model: {best}, Ortalama F1: {f1:.4f}")
    print(f"{target} - En iyi model: {best}, Ortalama ACC: {acc:.4f}")

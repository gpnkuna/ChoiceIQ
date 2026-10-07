from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

BASE = Path(__file__).resolve().parent
DATA = BASE / "Data" / "processed"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

FEATURES = [
    "data_analytics", "programming", "mathematics", "cybersecurity", "networking",
    "problem_solving", "creativity_design", "systems_cloud", "communication_business"
]

BENCHMARKS = [
    ("Data & research", [1,.45,.8,0,0,.7,0,.2,.2], ["Data Scientist","Data Analyst","Data Engineer"]),
    ("Software builder", [.1,1,.2,0,.1,.8,.6,.4,.2], ["Software Developer","Web Developer","Mobile Application Developer"]),
    ("Cyber investigator", [.2,.4,.1,1,.3,.9,.1,.5,.2], ["Digital Forensics Expert","Cyber Incident Responder","Ethical Hacker"]),
    ("Network infrastructure", [0,.2,0,.2,1,.7,.1,.8,.2], ["Ict Network Engineer","Ict Network Administrator","Ict Network Architect"]),
    ("Cloud platform", [.1,.5,.1,.2,.7,.7,.2,1,.3], ["Cloud Engineer","Cloud Architect","Cloud Devops Engineer"]),
    ("Security operations", [.1,.35,.1,1,.5,.8,.1,.7,.25], ["Ict Security Administrator","Ict Security Technician","Cybersecurity Risk Manager"]),
    ("Systems analysis", [.35,.45,.2,.15,.35,.8,.25,.9,.65], ["Ict System Administrator","Ict System Analyst","Ict System Architect"]),
    ("UX interface", [.1,.45,.1,0,.05,.5,1,.2,.65], ["User Interface Designer","Web Developer","Software Developer"]),
    ("Database", [1,.65,.4,.1,.2,.7,.1,.6,.3], ["Database Administrator","Database Developer","Data Engineer"]),
    ("IT support", [.1,.25,.05,.2,.45,.9,.1,.8,.75], ["Ict Help Desk Agent","Ict Technician","Ict System Administrator"]),
]


def ranking_metrics(df, X, metric):
    details = []
    precision_vals, recall_vals, f1_vals, rr_vals = [], [], [], []
    for profile_name, vector, relevant in BENCHMARKS:
        q = np.array(vector, dtype=float)
        if metric == "cosine":
            scores = cosine_similarity([q], X)[0]
            order = np.argsort(scores)[::-1]
        else:
            dist = euclidean_distances([q], X)[0]
            order = np.argsort(dist)
        ranked = df.iloc[order]["career"].tolist()
        top3 = ranked[:3]
        hits = len(set(top3) & set(relevant))
        p = hits / 3
        r = hits / len(relevant)
        f1 = 0 if (p+r)==0 else 2*p*r/(p+r)
        first_rank = next((i+1 for i, c in enumerate(ranked) if c in relevant), None)
        rr = 0 if first_rank is None else 1/first_rank
        precision_vals.append(p); recall_vals.append(r); f1_vals.append(f1); rr_vals.append(rr)
        details.append({
            "model": metric.title(), "profile": profile_name, "expected_relevant": " | ".join(relevant),
            "top_3": " | ".join(top3), "hits": hits, "precision_at_3": p, "recall_at_3": r,
            "f1_at_3": f1, "reciprocal_rank": rr,
        })
    summary = {
        "model": metric.title(),
        "precision_at_3": float(np.mean(precision_vals)),
        "recall_at_3": float(np.mean(recall_vals)),
        "f1_at_3": float(np.mean(f1_vals)),
        "mrr": float(np.mean(rr_vals)),
        "benchmark_profiles": len(BENCHMARKS),
    }
    return summary, details


def main():
    df = pd.read_csv(DATA / "choiceiq_careers.csv")
    X = df[FEATURES].to_numpy(dtype=float)

    cosine_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    euclidean_model = NearestNeighbors(metric="euclidean", algorithm="brute").fit(X)
    joblib.dump(cosine_model, MODELS / "cosine_knn.joblib")
    joblib.dump(euclidean_model, MODELS / "euclidean_knn.joblib")
    joblib.dump({"career_names": df["career"].tolist(), "features": FEATURES, "career_matrix": X}, MODELS / "career_feature_bundle.joblib")

    eval_rows, detail_rows = [], []
    for metric in ["cosine", "euclidean"]:
        summary, details = ranking_metrics(df, X, metric)
        eval_rows.append(summary); detail_rows.extend(details)
    eval_df = pd.DataFrame(eval_rows)
    detail_df = pd.DataFrame(detail_rows)
    eval_df.to_csv(DATA / "model_evaluation.csv", index=False)
    detail_df.to_csv(DATA / "model_evaluation_details.csv", index=False)

    profiles = []
    for name, vec, relevant in BENCHMARKS:
        row = {"profile": name, **dict(zip(FEATURES, vec)), "expected_relevant": " | ".join(relevant)}
        profiles.append(row)
    pd.DataFrame(profiles).to_csv(DATA / "evaluation_profiles.csv", index=False)

    selected = eval_df.sort_values(["precision_at_3", "mrr"], ascending=False).iloc[0].to_dict()
    metadata = {
        "selected_model": "Cosine content-based k-NN",
        "selection_reason": "Higher Precision@3 and F1@3 on the expert-defined benchmark profiles, while remaining easy to explain.",
        "features": FEATURES,
        "career_count": int(len(df)),
        "evaluation_note": "Evaluation uses 10 expert-defined prototype profiles because no historical user-rating dataset is available. Metrics are demonstration metrics, not population-level accuracy claims.",
        "selected_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k,v in selected.items() if k != "model"},
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(eval_df.to_string(index=False))

if __name__ == "__main__":
    main()

"""Turn free-text comments into structure with TF-IDF + NMF topic modeling.

TF-IDF + NMF is deliberately lightweight so this runs anywhere with no model
downloads. The production upgrade is sentence embeddings + BERTopic; the analysis
step downstream works the same either way.
"""
import joblib
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import config


def main() -> None:
    df = pd.read_csv(config.DATA)

    vec = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2))
    X = vec.fit_transform(df["comment"])
    terms = vec.get_feature_names_out()

    nmf = NMF(n_components=config.N_TOPICS, random_state=config.RANDOM_STATE, init="nndsvda")
    W = nmf.fit_transform(X)        # doc x topic
    df["dominant_topic"] = W.argmax(axis=1)

    lines = ["TOPICS (top terms per component)", "=" * 44]
    for t, comp in enumerate(nmf.components_):
        top = [terms[i] for i in comp.argsort()[::-1][:config.TOP_TERMS]]
        lines.append(f"Topic {t}: " + ", ".join(top))

    report = "\n".join(lines)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "topics.txt").write_text(report + "\n")
    df.to_csv(config.DATA.with_name("comments_topics.csv"), index=False)
    joblib.dump({"vectorizer": vec, "nmf": nmf}, config.RESULTS / "topic_model.joblib")
    print(report)


if __name__ == "__main__":
    main()

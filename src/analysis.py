"""Connect topics to an outcome.

Word counts alone don't tell you what matters. The question a people-analytics
partner actually asks is: which themes travel with higher or lower engagement?
This step ranks topics by the mean engagement of the comments where they dominate,
so the text is tied to a measured outcome rather than left as description.
"""
import pandas as pd

import config


def main() -> None:
    df = pd.read_csv(config.DATA.with_name("comments_topics.csv"))

    summary = (df.groupby("dominant_topic")
                 .agg(n=("engagement", "size"),
                      mean_engagement=("engagement", "mean"))
                 .sort_values("mean_engagement", ascending=False))

    overall = df["engagement"].mean()
    lines = ["TOPIC x ENGAGEMENT", "=" * 44,
             f"Overall mean engagement: {overall:.2f}", "",
             f"{'topic':<8}{'n':>6}{'mean':>10}{'vs overall':>14}"]
    for topic, row in summary.iterrows():
        delta = row["mean_engagement"] - overall
        lines.append(f"{int(topic):<8}{int(row['n']):>6}{row['mean_engagement']:>10.2f}{delta:>+14.2f}")

    lines += ["",
              "Pair this with topics.txt: the highest- and lowest-engagement topics",
              "are the themes worth acting on, not just the most frequent words."]

    report = "\n".join(lines)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "topic_engagement.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()

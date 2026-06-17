# employee-comments-nlp
“ Turn open-ended survey comments into themes, then link each theme to engagement — what to act on, not a word cloud.”
# What Employees Actually Say

**Turning open-ended survey comments into themes, then tying those themes to engagement.**

Open-ended survey questions collect the richest feedback a company has and then usually
waste it in a word cloud. This project takes free-text employee comments, finds the themes
inside them, and connects each theme to a measured outcome, so the result is a short list
of what to act on rather than a picture of the most common words.

---

## What it does

```
comments ──▶ TF-IDF + NMF ──▶ dominant theme ──▶ theme x engagement ──▶ report
(free text)   topic model      per comment        which themes track
                                                   higher / lower engagement
```

- **Structure from text.** TF-IDF features feed an NMF topic model that surfaces the latent
  themes (growth, workload, recognition, direction, pay) and labels each comment with its
  dominant theme.
- **Themes linked to an outcome.** Each theme is ranked by the mean engagement of the
  comments where it dominates, turning description into something decision-makers can use.
- **Lightweight on purpose.** TF-IDF + NMF runs anywhere with no model downloads. The
  production upgrade is sentence embeddings + BERTopic; the analysis step is identical
  either way.

The bundled comments are synthetic, drawn from themes with planted engagement pulls, so the
model has real structure to recover.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate_data   # synthetic comments + engagement scores
python -m src.topics          # TF-IDF + NMF topics
python -m src.analysis        # theme x engagement
```

Outputs land in `results/`.

---

## Why this framing

A word cloud tells you "workload" appears often. It does not tell you whether the people
writing about workload are the disengaged ones. Connecting each theme to an engagement
outcome is the difference between *interesting* and *actionable*, and it mirrors how a
people-analytics function actually prioritizes: by what moves the outcome, not by frequency.

---

## Repository structure

```
employee-comments-nlp/
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── generate_data.py    # synthetic comments tied to engagement
│   ├── topics.py           # TF-IDF + NMF topic model
│   └── analysis.py         # theme x engagement summary
├── data/                   # generated data (git-ignored)
└── results/                # topic + engagement reports (git-ignored)
```

## Limitations and next steps

- Synthetic comments are short and clean; real survey text is messier and would need more
  preprocessing and a larger vocabulary.
- Sentence embeddings + BERTopic would capture meaning that bag-of-words misses.
- A natural extension is modeling engagement from theme prevalence with proper inference
  (confidence intervals, controls) rather than group means alone.

## License

MIT.

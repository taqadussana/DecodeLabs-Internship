# Project 3 — AI Recommendation Logic (Tech Stack Recommender)

Third milestone of the DecodeLabs AI Industrial Training Kit (Batch 2026).
Content-based filtering engine that maps a user's raw skills to the
closest-matching job roles.

## Goal
Create a simple recommendation system based on user preferences:
take user input, match it against a dataset using similarity logic,
and display the top recommended items.

## Approach: Content-Based Filtering
Project 3 deliberately uses content-based filtering (item attributes)
rather than collaborative filtering (community behavior), since it works
immediately without needing a large historical interaction dataset.

## Pipeline (IPO Framework, 4-step ranking pipeline)
| Step | What happens |
|---|---|
| **1. Ingestion** | Capture at least 3 user skills as raw text input |
| **2. Scoring** | Vectorize the user's skills and every job role's skills with **TF-IDF** (shared vocabulary space), then compute **Cosine Similarity** between the user vector and each role vector |
| **3. Sorting** | Rank all roles by descending similarity score |
| **4. Filtering** | Truncate to the **Top 3** roles to prevent choice overload |

## Why TF-IDF + Cosine Similarity (not raw overlap or Euclidean distance)
- **TF-IDF** weights specific/descriptive skills (e.g. "Kubernetes") higher
  than generic ones, instead of treating every matching tag equally like a
  plain binary/Jaccard overlap would.
- **Cosine similarity** measures the *angle* between vectors, not their
  raw magnitude — so a user who lists 3 skills isn't unfairly penalized
  against a job role with a longer skill list. Euclidean distance would be
  sensitive to that size difference; cosine similarity isn't.

## Files
- `raw_skills.csv` — dataset: 12 job roles, each with its associated skills
- `recommender.py` — core pipeline (`recommend()`), runnable with example inputs
- `interactive.py` — CLI wrapper that prompts the user for skills at runtime
- `requirements.txt` — dependencies

## Run it
```bash
pip install -r requirements.txt

# quick demo with hardcoded example skills
python recommender.py

# interactive version — type your own skills
python interactive.py
```

## Cold Start
If a user enters skills that don't appear anywhere in `raw_skills.csv`
(e.g. "Cooking", "Painting"), every similarity score comes back as 0% —
this is the classic **Cold Start problem** from the deck: no shared
vocabulary means no similarity math to run. `recommender.py` includes a
demo of this at the bottom of its `__main__` block.

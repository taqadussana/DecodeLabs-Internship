"""
Project 3 – AI Recommendation Logic: Tech Stack Recommender
DecodeLabs Industrial Training Kit (2026)

Goal: Map a user's raw skills to the closest-matching job roles using
content-based filtering (TF-IDF vectorization + Cosine Similarity).

Pipeline (IPO Framework, 4-step ranking pipeline from the deck):
  1. INGESTION -> capture user skills (min 3 inputs)
  2. SCORING   -> TF-IDF vectorize + cosine similarity vs. every job role
  3. SORTING   -> rank roles by descending similarity score
  4. FILTERING -> truncate to Top-N (Top 3) to avoid choice overload
"""

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(path="raw_skills.csv"):
    """Load job roles and their skill sets from the CSV 'item' dataset."""
    roles, skill_docs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roles.append(row["role"])
            skill_docs.append(row["skills"])
    return roles, skill_docs


def ingest_user_skills(skills):
    """
    STEP 1: INGESTION
    Capture the user state as a minimum of three skills, joined into
    a single 'document' so it can be vectorized in the same space
    as the job-role documents (the shared vocabulary requirement).
    """
    if len(skills) < 3:
        raise ValueError("Project 3 requires at least 3 user skill inputs.")
    return " ".join(skills)


def score_roles(roles, skill_docs, user_doc):
    """
    STEP 2: SCORING
    Fit TF-IDF across the job-role documents + the user's document so
    they share one vocabulary space, then compute cosine similarity
    between the user vector and every role vector.
    """
    corpus = skill_docs + [user_doc]

    vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z\+\#\.]+")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    role_vectors = tfidf_matrix[:-1]   # all rows except the last
    user_vector = tfidf_matrix[-1]     # the last row is the user

    scores = cosine_similarity(user_vector, role_vectors).flatten()
    return list(zip(roles, scores))


def sort_and_filter(scored_roles, top_n=3):
    """
    STEP 3: SORTING   -> descending order by score
    STEP 4: FILTERING -> truncate to top_n to prevent choice overload
    """
    ranked = sorted(scored_roles, key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]


def recommend(user_skills, dataset_path="raw_skills.csv", top_n=3):
    """Run the full 4-step pipeline and return the Top-N recommendations."""
    roles, skill_docs = load_dataset(dataset_path)
    user_doc = ingest_user_skills(user_skills)
    scored = score_roles(roles, skill_docs, user_doc)
    return sort_and_filter(scored, top_n=top_n)


def print_recommendations(user_skills, results):
    print("=" * 60)
    print("TECH STACK RECOMMENDER")
    print("=" * 60)
    print(f"Input skills: {user_skills}\n")
    print(f"Top {len(results)} matching career paths:\n")
    for rank, (role, score) in enumerate(results, start=1):
        pct = score * 100
        print(f"{rank}. {role:<25} match: {pct:5.1f}%")


if __name__ == "__main__":
    # Example run — swap this list for interactive input() calls if needed
    example_skills = ["Python", "Cloud", "Automation"]
    results = recommend(example_skills)
    print_recommendations(example_skills, results)

    print("\n" + "-" * 60)
    print("Cold-start check (0 shared vocabulary terms):")
    cold_skills = ["Cooking", "Painting", "Gardening"]
    cold_results = recommend(cold_skills)
    print_recommendations(cold_skills, cold_results)

"""
Project 3 – Interactive runner.
Prompts the user for at least 3 skills, then prints the Top-3 recommended
job roles using the same recommend() pipeline from recommender.py.
"""

from recommender import recommend, print_recommendations


def get_user_skills(min_skills=3):
    print("Enter your skills one at a time (press Enter with nothing typed to finish).")
    print(f"You need at least {min_skills} skills.\n")

    skills = []
    while True:
        skill = input(f"Skill #{len(skills) + 1}: ").strip()
        if not skill:
            if len(skills) >= min_skills:
                break
            print(f"Please enter at least {min_skills} skills before finishing.")
            continue
        skills.append(skill)
    return skills


if __name__ == "__main__":
    user_skills = get_user_skills()
    results = recommend(user_skills)
    print()
    print_recommendations(user_skills, results)

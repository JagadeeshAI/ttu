#!/usr/bin/env python3
"""
Generate 100 realistic, culturally grounded synthetic profiles (India & USA)
with private, personal, and social details.
Each country's personas are matched to correct national political parties.
"""

import json
import random
from faker import Faker
from datetime import date, timedelta

# ---------------- CONFIG ----------------
NUM_PROFILES = 100
SEED = 42
OUTPUT_FILE = "data/data.json"

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)
fake.seed_instance(SEED)

# ---------------- COUNTRY DATA ----------------
COUNTRIES = ["India", "United States"]

# Names
INDIAN_FIRST_NAMES = ["Aarav", "Vihaan", "Arjun", "Ishaan", "Raj", "Kiran", "Priya", "Ananya", "Sneha", "Kavya"]
INDIAN_LAST_NAMES = ["Sharma", "Patel", "Reddy", "Nair", "Gupta", "Mehta", "Rao", "Singh", "Chopra", "Iyer"]
US_FIRST_NAMES = ["John", "Emily", "Michael", "Sarah", "David", "Ashley", "James", "Olivia", "Daniel", "Jessica"]
US_LAST_NAMES = ["Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Taylor", "Anderson", "Clark", "Hall"]

# Locations
INDIAN_STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Kerala", "Delhi", "Gujarat", "Telangana", "Punjab"]
US_STATES = ["California", "Texas", "New York", "Florida", "Illinois", "Washington", "Ohio", "Georgia"]
INDIAN_CITIES = ["Mumbai", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Ahmedabad", "Kochi", "Pune"]
US_CITIES = ["San Francisco", "Austin", "New York City", "Chicago", "Seattle", "Miami", "Boston", "Atlanta"]

# Religion & Politics
RELIGIONS_INDIA = ["Hinduism", "Islam", "Christianity", "Sikhism", "Buddhism"]
RELIGIONS_USA = ["Christianity", "Judaism", "Islam", "Atheism", "Agnostic"]
PARTIES_INDIA = ["Bharatiya Janata Party", "Indian National Congress", "Aam Aadmi Party"]
PARTIES_USA = ["Democratic Party", "Republican Party", "Libertarian Party"]

# Education
DEGREES = ["B.Tech", "M.Tech", "MBA", "Ph.D.", "M.Sc.", "B.Sc.", "BA", "B.Com"]
UNIS_INDIA = ["IIT Delhi", "IIT Madras", "IIT Bombay", "IIT Mandi", "NIT Trichy", "BITS Pilani"]
UNIS_USA = ["Harvard University", "Stanford University", "MIT", "UC Berkeley", "Carnegie Mellon University", "Yale University"]

# Hobbies, Books, Traits
HOBBIES = ["reading", "traveling", "music", "photography", "sports", "cooking", "painting", "gaming"]
BOOKS = ["The Art of Peace", "Sapiens", "The Alchemist", "Atomic Habits", "Meditations", "The Power of Now", "Educated"]
TRAITS = ["disciplined", "analytical", "calm", "curious", "empathetic", "creative", "confident", "humble"]
PETS = ["dog", "cat", "parrot", "rabbit", "none"]

# Movies
MOVIE_GENRES_INDIA = {
    "Action": ["War", "Baahubali", "KGF", "Vikram", "Pathaan"],
    "Drama": ["3 Idiots", "Article 15", "Dangal", "Taare Zameen Par"],
    "Comedy": ["Hera Pheri", "Welcome", "Chupke Chupke", "Golmaal"],
    "Romance": ["DDLJ", "Yeh Jawaani Hai Deewani", "Kabir Singh"]
}
MOVIE_GENRES_USA = {
    "Action": ["The Dark Knight", "John Wick", "Mad Max: Fury Road"],
    "Drama": ["The Shawshank Redemption", "Forrest Gump", "The Godfather"],
    "Comedy": ["Superbad", "The Hangover", "Zombieland"],
    "Romance": ["La La Land", "The Notebook", "Pride and Prejudice"]
}

# ---------------- PARTY-SPECIFIC POLITICAL VIEWS ----------------
VIEWS_SUMMARY = {
    # INDIA
    "Bharatiya Janata Party": "Believes in innovation, national unity, and self-reliance through transparent, technology-driven governance.",
    "Indian National Congress": "Supports inclusive welfare, women’s empowerment, and institutional revival for a socially just democracy.",
    "Aam Aadmi Party": "Believes in citizen-first governance and decentralized decision-making for transparency and equality.",
    # USA
    "Democratic Party": "Advocates equality, climate action, and inclusive opportunity through progressive policies.",
    "Republican Party": "Prioritizes economic freedom, entrepreneurship, and national security under a conservative framework.",
    "Libertarian Party": "Champions personal liberty and minimal government interference in social and economic life."
}

REFORM_FOCUS = {
    # INDIA
    "Bharatiya Janata Party": [
        "Promote Make-in-India manufacturing and entrepreneurship.",
        "Build knowledge-based economic hubs and smart infrastructure.",
        "Launch clean air and green mobility programs.",
        "Strengthen digital transparency and governance.",
        "Advance youth skill development in technology sectors."
    ],
    "Indian National Congress": [
        "Expand universal healthcare and pension coverage.",
        "Launch income support programs for women and workers.",
        "Revive participatory institutions in education and health.",
        "Promote cultural inclusion and minority welfare.",
        "Ensure fair wages and labor regularization."
    ],
    "Aam Aadmi Party": [
        "Decentralize decision-making via Mohalla Sabhas.",
        "Provide free quality healthcare and school education.",
        "Guarantee water, electricity, and safety for all.",
        "Empower informal workers and local vendors.",
        "Seek statehood to ensure Delhi’s autonomy."
    ],
    # USA
    "Democratic Party": [
        "Expand renewable energy jobs and green infrastructure.",
        "Ensure healthcare affordability and student debt relief.",
        "Advance diversity and workplace equality.",
        "Regulate big tech for ethical innovation.",
        "Strengthen community-based economic programs."
    ],
    "Republican Party": [
        "Encourage small-business deregulation and tax reform.",
        "Strengthen defense innovation and veteran support.",
        "Expand domestic energy production and jobs.",
        "Support traditional family values and self-reliance.",
        "Promote entrepreneurship and free-market competitiveness."
    ],
    "Libertarian Party": [
        "Reduce federal bureaucracy and government oversight.",
        "Expand civil liberties and privacy rights.",
        "Encourage cryptocurrency and free markets.",
        "Promote voluntary welfare and private charity.",
        "Limit military intervention abroad."
    ]
}

# ---------------- HELPERS ----------------
def make_name(country):
    if country == "India":
        return f"{random.choice(INDIAN_FIRST_NAMES)} {random.choice(INDIAN_LAST_NAMES)}"
    else:
        return f"{random.choice(US_FIRST_NAMES)} {random.choice(US_LAST_NAMES)}"

def random_dob(min_age=22, max_age=60):
    today = date.today()
    start = today.replace(year=today.year - max_age)
    end = today.replace(year=today.year - min_age)
    dob = start + timedelta(days=random.randint(0, (end - start).days))
    return dob.strftime("%Y-%m-%d")

def random_passport(country):
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000,9999999)}"

def random_phone(country):
    if country == "India":
        return f"+91{random.randint(6000000000,9999999999)}"
    else:
        return f"+1{random.randint(2000000000,9999999999)}"

def random_email(name, country):
    domains = ["gmail.com", "yahoo.com", "outlook.com"]
    suffix = ".in" if country == "India" else ".com"
    local = name.lower().replace(" ", ".")
    return f"{local}{random.randint(10,99)}@{random.choice(domains).replace('.com', suffix)}"

def random_national_id(country):
    return f"{'AXP' if country == 'India' else 'US'}{random.randint(100000000,999999999)}"

def random_address(country):
    if country == "India":
        return {"city": random.choice(INDIAN_CITIES), "state": random.choice(INDIAN_STATES), "country": "India"}
    else:
        return {"city": random.choice(US_CITIES), "state": random.choice(US_STATES), "country": "United States"}

def random_education(country):
    degree = random.choice(DEGREES)
    uni = random.choice(UNIS_INDIA if country == "India" else UNIS_USA)
    field = random.choice(["Computer Science", "Economics", "Physics", "Engineering", "Psychology", "Literature"])
    return f"{degree} in {field}, {uni}"

def random_movie_preferences(country):
    if country == "India":
        genre = random.choice(list(MOVIE_GENRES_INDIA.keys()))
        movies = random.sample(MOVIE_GENRES_INDIA[genre], 2)
    else:
        genre = random.choice(list(MOVIE_GENRES_USA.keys()))
        movies = random.sample(MOVIE_GENRES_USA[genre], 2)
    return genre, movies

def random_language(country):
    if country == "India":
        return random.sample(["Hindi", "English", "Tamil", "Telugu", "Kannada", "Malayalam"], 3)
    else:
        return random.sample(["English", "Spanish", "French"], 2)

def random_social(country, name, party):
    profile = f"linkedin.com/in/{name.lower().replace(' ', '')}"
    volunteering = ["teaching underprivileged children art", "community mentorship", "animal shelter volunteering"]
    fav_travel = random.choice(["Sikkim", "Goa", "Leh", "Rajasthan", "Colorado", "Hawaii", "New York"])

    return {
        "public_profile": profile,
        "volunteering": random.sample(volunteering, 1),
        "political_vision": {
            "party": party,
            "views_summary": VIEWS_SUMMARY.get(party, "Holds moderate and pragmatic beliefs."),
            "if_leader_reforms": REFORM_FOCUS.get(party, [])
        },
        "friends_description": random.choice(["Calm yet determined", "Creative and kind", "Disciplined but humorous", "Empathetic and focused"]),
        "fav_travel_destination": fav_travel,
        "tag": "social"
    }

# ---------------- MAIN ----------------
profiles = []
for _ in range(NUM_PROFILES):
    country = random.choice(COUNTRIES)
    name = make_name(country)
    dob = random_dob()
    passport = random_passport(country)
    phone = random_phone(country)
    email = random_email(name, country)
    national_id = random_national_id(country)
    edu = random_education(country)
    address = random_address(country)
    religion = random.choice(RELIGIONS_INDIA if country == "India" else RELIGIONS_USA)
    fav_party = random.choice(PARTIES_INDIA if country == "India" else PARTIES_USA)
    fav_genre, fav_movies = random_movie_preferences(country)
    hobbies = random.sample(HOBBIES, 2)
    traits = random.sample(TRAITS, 3)
    languages = random_language(country)
    pet = random.choice(PETS)
    fav_book = random.choice(BOOKS)
    life_goal = random.choice([
        "To promote peace through creativity",
        "To lead innovative projects that help society",
        "To inspire others through storytelling and mentorship",
        "To explore the intersection of technology and humanity"
    ])
    philosophy = random.choice([
        "Balance between strength and serenity",
        "Kindness is the highest form of intelligence",
        "Discipline and curiosity build a meaningful life"
    ])

    profile = {
        "private": {
            "name_full": name,
            "dob": dob,
            "passport_number": passport,
            "phone_number": phone,
            "email": email,
            "national_id": national_id,
            "tag": "private"
        },
        "personal": {
            "education_details": edu,
            "fav_political_party": fav_party,
            "religion": religion,
            "hobbies": hobbies,
            "fav_movie_genre": fav_genre,
            "fav_movies": fav_movies,
            "fav_book": fav_book,
            "personality_traits": traits,
            "address": address,
            "languages_known": languages,
            "pet": pet,
            "life_goal": life_goal,
            "philosophy": philosophy,
            "tag": "personal"
        },
        "social": random_social(country, name, fav_party)
    }
    profiles.append(profile)

# ---------------- OUTPUT ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False)

print(f"🎯 Generated {len(profiles)} extended, country-consistent profiles → {OUTPUT_FILE}")

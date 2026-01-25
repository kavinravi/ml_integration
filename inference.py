import pandas as pd
import joblib

model = joblib.load("xgb_churn.joblib")

# Collect user input
# Define valid options
genres = ["Comedy", "Thriller", "ScienceFiction", "RomanticComedy", "Drama", "Horror", "Action", "Documentary"]
plans = ["A", "B", "P"]
genders = ["woman", "man", "nonbinary", "other"]
yes_no = ["No", "Yes"]

def choose(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    choice = int(input("Enter number: "))
    return options[choice - 1]

def main():
    # Use it
    gender = choose("Gender:", genders)
    plan = choose("Plan:", plans)
    topgenre = choose("Top genre:", genres)
    secondgenre = choose("Second genre:", genres)
    age = float(input("Age: "))
    income = float(input("Income: "))
    monthssubbed = int(input("Months subscribed: "))
    meanhourswatched = float(input("Mean hours watched: "))
    competitorsub = yes_no.index(choose("Has competitor subscription?", yes_no))
    numprofiles = int(input("Number of profiles: "))
    cancelled = float(yes_no.index(choose("Cancelled before?", yes_no)))
    downgraded = yes_no.index(choose("Downgraded before?", yes_no))
    bundle = yes_no.index(choose("Has bundle?", yes_no))
    kids = yes_no.index(choose("Has kids content?", yes_no))
    longestsession = float(input("Longest session (hours): "))

    # Build DataFrame
    user_input = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "income": [income],
        "monthssubbed": [monthssubbed],
        "plan": [plan],
        "meanhourswatched": [meanhourswatched],
        "competitorsub": [competitorsub],
        "numprofiles": [numprofiles],
        "cancelled": [cancelled],
        "downgraded": [downgraded],
        "bundle": [bundle],
        "kids": [kids],
        "longestsession": [longestsession],
        "topgenre": [topgenre],
        "secondgenre": [secondgenre]
    })

    # Predict
    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0, 1]

    print(f"\nPrediction: {'Will churn' if pred == 1 else 'Will not churn'}")
    print(f"Churn probability: {prob:.1%}")

if __name__ == "__main__":
    main()
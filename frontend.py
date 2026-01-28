import gradio as gr
import pandas as pd
import joblib

model = joblib.load("xgb_churn.joblib")

GENDER_CHOICES = ["woman", "man", "nonbinary", "other"]
PLAN_CHOICES = ["A", "B", "P"]
GENRE_CHOICES = ["Comedy", "Thriller", "ScienceFiction", "RomanticComedy", "Drama", "Horror", "Action", "Documentary"]


def predict_churn(
    gender: str,
    age: float,
    income: float,
    months_subbed: int,
    plan: str,
    mean_hours_watched: float,
    competitor_sub: bool,
    num_profiles: int,
    cancelled: bool,
    downgraded: bool,
    bundle: bool,
    kids: bool,
    longest_session: float,
    top_genre: str,
    second_genre: str,
) -> tuple[str, float]:
    """Predict customer churn based on input features."""
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "income": [income],
        "monthssubbed": [months_subbed],
        "plan": [plan],
        "meanhourswatched": [mean_hours_watched],
        "competitorsub": [int(competitor_sub)],
        "numprofiles": [num_profiles],
        "cancelled": [float(cancelled)],
        "downgraded": [int(downgraded)],
        "bundle": [int(bundle)],
        "kids": [int(kids)],
        "longestsession": [longest_session],
        "topgenre": [top_genre],
        "secondgenre": [second_genre],
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        label = f"High Churn Risk ({probability:.1%} probability)"
    else:
        label = f"Low Churn Risk ({probability:.1%} probability)"

    return label, probability


with gr.Blocks(title="Customer Churn Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Customer Churn Prediction
        
        Predict whether a streaming service customer is likely to churn based on their profile and behavior.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Demographics")
            gender = gr.Dropdown(choices=GENDER_CHOICES, value="woman", label="Gender")
            age = gr.Slider(minimum=18, maximum=100, value=35, step=1, label="Age")
            income = gr.Number(value=50000, label="Annual Income ($)")

        with gr.Column():
            gr.Markdown("### Subscription Details")
            plan = gr.Dropdown(choices=PLAN_CHOICES, value="B", label="Subscription Plan")
            months_subbed = gr.Slider(minimum=0, maximum=120, value=12, step=1, label="Months Subscribed")
            bundle = gr.Checkbox(value=False, label="Has Bundle")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Viewing Behavior")
            mean_hours_watched = gr.Number(value=20, label="Mean Hours Watched (per month)")
            longest_session = gr.Number(value=4, label="Longest Session (hours)")
            num_profiles = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of Profiles")

        with gr.Column():
            gr.Markdown("### Content Preferences")
            top_genre = gr.Dropdown(choices=GENRE_CHOICES, value="Comedy", label="Top Genre")
            second_genre = gr.Dropdown(choices=GENRE_CHOICES, value="Drama", label="Second Genre")
            kids = gr.Checkbox(value=False, label="Has Kids Content")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Account Status")
            competitor_sub = gr.Checkbox(value=False, label="Has Competitor Subscription")
            cancelled = gr.Checkbox(value=False, label="Previously Cancelled")
            downgraded = gr.Checkbox(value=False, label="Previously Downgraded")

    with gr.Row():
        predict_btn = gr.Button("Predict Churn", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            probability_output = gr.Slider(
                minimum=0,
                maximum=1,
                value=0,
                label="Churn Probability",
                interactive=False,
            )

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender,
            age,
            income,
            months_subbed,
            plan,
            mean_hours_watched,
            competitor_sub,
            num_profiles,
            cancelled,
            downgraded,
            bundle,
            kids,
            longest_session,
            top_genre,
            second_genre,
        ],
        outputs=[prediction_output, probability_output],
    )

if __name__ == "__main__":
    demo.launch()

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import os
print("KEY:", os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Initialize OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_text = data.get("text")

        if not user_text:
            return jsonify({"response": "No input text received"}), 400

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis assistant. Respond in JSON only."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the sentiment of this text and respond ONLY in JSON.

Text: "{user_text}"

Format:
{{
  "sentiment": "POSITIVE | NEGATIVE | NEUTRAL",
  "confidence": "percentage",
  "reason": "short explanation"
}}
"""
                }
            ]
        )

        reply = completion.choices[0].message.content

        return jsonify({"response": reply})

    except Exception as e:
        # VERY IMPORTANT: prevents frontend crash
        return jsonify({
            "response": f"Server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

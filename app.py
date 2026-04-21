from flask import Flask, jsonify, render_template, request
from chatbot import get_response

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_input = (data.get("message") or "").strip()

        if not user_input:
            return jsonify({"response": "Please type a message."}), 400

        # This now calls the Hybrid logic (Local ML first, then LLM)
        response = get_response(user_input)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    # Host 0.0.0.0 is essential for Presidency University network access
    app.run(host="0.0.0.0", port=5000, debug=True)

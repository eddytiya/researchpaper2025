import os
os.environ["H2O_DISABLE_XGBOOST"] = "TRUE"

import h2o
from flask import Flask, request, jsonify, render_template
import pandas as pd

# Initialize H2O
h2o.init()

# Load saved H2O model
model_path = "models/StackedEnsemble_BestOfFamily_1_AutoML_2_20250928_95852"
model = h2o.load_model(model_path)

# Get model columns (excluding target)
model_columns = [col for col in model._model_json['output']['names'] if col != 'Survived']

# Flask app
app = Flask(__name__)

# Define which columns are categorical in your model
categorical_cols = ['Pclass', 'Sex', 'Embarked']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        df = pd.DataFrame(data)

        # Add missing columns (skip Survived)
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0  # numeric default

        # Reorder columns to match model
        df = df[model_columns]

        # Ensure categorical columns are strings
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Ensure numeric columns are floats
        for col in df.columns:
            if col not in categorical_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert to H2OFrame
        hf = h2o.H2OFrame(df)

        # Predict
        preds = model.predict(hf)

        # Return predictions as JSON
        return jsonify(preds.as_data_frame().to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


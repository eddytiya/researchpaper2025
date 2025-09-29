import h2o

# Initialize H2O
h2o.init()

# Load your saved model
model_path = "models/StackedEnsemble_BestOfFamily_1_AutoML_2_20250928_95852"
model = h2o.load_model(model_path)

# Print expected column names
print("Expected columns for prediction:")
print(model._model_json["output"]["names"])

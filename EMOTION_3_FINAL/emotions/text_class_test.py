import joblib

# Load the model
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))

# Check the class names if available
if hasattr(pipe_lr, "classes_"):
    print("Class names:", pipe_lr.classes_)
else:
    print("Class names not available.")

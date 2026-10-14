Place your trained model here as `best_model.keras`.

Optionally add a `class_names.json` file (a JSON list of class labels, in
the exact order your model outputs them) to override the default classes
defined at the top of app.py:

    ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

Getting the order wrong is the single most common cause of "the model
seems accurate but the label is wrong" — index 0 in class_names.json (or
CLASS_NAMES in app.py) must match whatever class your model puts first
in its output vector.

If no model file is present, app.py does NOT fake a result. It loads at
startup, logs a clear error to the terminal, shows a warning banner on
the Dashboard and New Analysis pages, and blocks new analyses until a
valid model is in place. This is intentional — silently returning fake
predictions would be worse than refusing to run.

## Grad-CAM

Every analysis also generates a Grad-CAM heatmap (shown on the result
page next to the original image) showing which parts of the image the
model focused on for its prediction. This requires your model to have at
least one Conv2D-style layer.

If Grad-CAM can't be generated for your specific architecture (e.g. no
convolutional layers, or an incompatible graph structure), the result
page still shows your prediction normally — it just displays a clear
"Grad-CAM unavailable — technical issue" notice with the actual reason,
instead of failing the whole analysis or showing nothing.

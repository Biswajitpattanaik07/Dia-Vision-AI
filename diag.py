"""
DiaVision AI — Flask application
=================================
Diabetic retinopathy screening dashboard backed by your trained Keras
model at models/best_model.keras.

IMPORTANT — before running:
1.  Drop your trained model at:  models/best_model.keras
2.  Edit CLASS_NAMES below so it EXACTLY matches your model's output
    order (index 0 = first class your model predicts, etc). Get this
    wrong and predictions will map to the wrong label even though the
    model itself is correct.

Run:
    python -m pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
Demo login: doctor@diavision.ai / password123
"""
import json
import os
from datetime import datetime

from flask import (Flask, render_template, redirect, url_for, request,
                    flash, send_from_directory, jsonify)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                          login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads TURSO_DATABASE_URL, GROK_API_KEY, etc. from a local .env file
except ImportError:
    pass  # python-dotenv not installed — env vars can still be set directly on the OS

# ---------------------------------------------------------------------------
# App / config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORTS_FOLDER"] = REPORTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads

# --- Database: Turso (libSQL) if configured, otherwise local SQLite -------
# Set TURSO_DATABASE_URL (e.g. "libsql://your-db-org.turso.io") and
# TURSO_AUTH_TOKEN in a .env file (see .env.example) to use Turso.
# Requires: pip install sqlalchemy-libsql
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
USING_TURSO = False


def _turso_connection_works(uri):
    """Quick pre-flight connectivity check. Without this, a wrong token,
    a typo'd hostname, or Turso being temporarily unreachable would crash
    the entire app at startup with a raw traceback the first time a route
    touches the database — instead we test once up front and fall back
    to local SQLite with a clear message if it fails."""
    try:
        from sqlalchemy import create_engine, text
        test_engine = create_engine(uri)
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        test_engine.dispose()
        return True, None
    except Exception as exc:
        return False, str(exc)


if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
    try:
        import sqlalchemy_libsql  # noqa: F401  (registers the sqlite+libsql dialect)
        turso_uri = TURSO_DATABASE_URL.replace("libsql://", "sqlite+libsql://")
        sep = "&" if "?" in turso_uri else "?"
        turso_uri = f"{turso_uri}{sep}authToken={TURSO_AUTH_TOKEN}&secure=true"

        ok, err = _turso_connection_works(turso_uri)
        if ok:
            app.config["SQLALCHEMY_DATABASE_URI"] = turso_uri
            USING_TURSO = True
            print(f"[DiaVision] Using Turso database: {TURSO_DATABASE_URL}")
        else:
            print(f"[DiaVision] Could not connect to Turso ({err}). Falling back to local SQLite. "
                  f"Check TURSO_DATABASE_URL / TURSO_AUTH_TOKEN in .env, and your network egress "
                  f"settings if you're behind a firewall.")
    except ImportError:
        print("[DiaVision] TURSO_DATABASE_URL is set but 'sqlalchemy-libsql' isn't installed "
              "(pip install sqlalchemy-libsql). Falling back to local SQLite.")

if not USING_TURSO:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "diavision.db")
    print("[DiaVision] Using local SQLite database (diavision.db)")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

# ---------------------------------------------------------------------------
# >>> EDIT THIS to match your model's output classes, in order. <<<
# Index 0 must be whatever class your model puts first in its output
# vector, index 1 the second, etc. If unsure, check how you set up your
# training generator / dataset — the class order it used is the order
# your model predicts in.
# ---------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Optional: instead of editing the list above, you can drop a
# models/class_names.json file (a JSON list of strings, in model output
# order) and it will be used automatically instead.
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = json.load(f)


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(60), default="Clinician")
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_code = db.Column(db.String(40), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(30))
    diabetes_type = db.Column(db.String(30))
    duration_years = db.Column(db.Float)
    city = db.Column(db.String(120))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    analyses = db.relationship("Analysis", backref="patient", lazy=True,
                                order_by="desc(Analysis.created_at)")

    @property
    def total_screenings(self):
        return len(self.analyses)

    @property
    def latest_analysis(self):
        return self.analyses[0] if self.analyses else None


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    prediction_label = db.Column(db.String(60), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities_json = db.Column(db.Text, nullable=False)  # JSON: {label: prob}
    reviewed_by = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Grad-CAM: filename of the generated heatmap overlay, if it could be
    # produced. gradcam_error holds a human-readable technical reason if
    # generation failed — the prediction itself still succeeds either way.
    gradcam_filename = db.Column(db.String(255))
    gradcam_error = db.Column(db.Text)

    @property
    def probabilities(self):
        return json.loads(self.probabilities_json)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------------------------------------------------
# Model loading — loaded ONCE at startup, for real. If this fails, the
# app still starts (so you can see the dashboard etc.) but every analysis
# attempt will show a clear error telling you exactly what's wrong,
# instead of silently guessing.
# ---------------------------------------------------------------------------
MODEL = None
MODEL_LOAD_ERROR = None

try:
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        MODEL_LOAD_ERROR = (
            f"No model file found at {MODEL_PATH}. "
            f"Copy your trained model there as 'best_model.keras'."
        )
    else:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"[DiaVision] Loaded model from {MODEL_PATH}")
        print(f"[DiaVision] Model input shape: {MODEL.input_shape}")
        print(f"[DiaVision] Model output units: {MODEL.output_shape}")
        print("[DiaVision] Explanation code version: v8 (EfficientNet raw-pixel mode)")
        print(f"[DiaVision] Preprocess mode: {PREPROCESS_MODE}")
        print("[DiaVision] Model layers:")
        for _i, _l in enumerate(MODEL.layers):
            print(f"    {_i:>3}  {_l.name}  ({type(_l).__name__})")
        if MODEL.output_shape[-1] != len(CLASS_NAMES):
            print(
                f"[DiaVision] WARNING: model outputs {MODEL.output_shape[-1]} classes "
                f"but CLASS_NAMES in app.py has {len(CLASS_NAMES)} entries. "
                f"Fix CLASS_NAMES so it matches, or predictions will be mislabeled."
            )
except ImportError:
    MODEL_LOAD_ERROR = (
        "TensorFlow isn't installed. Run: python -m pip install tensorflow"
    )
except Exception as exc:
    MODEL_LOAD_ERROR = f"Failed to load model from {MODEL_PATH}: {exc}"

if MODEL_LOAD_ERROR:
    print(f"[DiaVision] MODEL NOT LOADED — {MODEL_LOAD_ERROR}")


# ---------------------------------------------------------------------------
# >>> PREPROCESSING — must match how the model was TRAINED. <<<
#   "raw"   : pixels stay 0-255 (use for EfficientNet/ResNet-style models that
#             have their own scaling built in, or were trained on raw pixels)
#   "scale" : pixels divided by 255 -> 0-1 (use if you trained with rescale=1/255)
# If Grad-CAM lights up the image corners/background, try the other option.
# ---------------------------------------------------------------------------
PREPROCESS_MODE = os.environ.get("PREPROCESS_MODE", "raw").strip().lower()


def preprocess_pil(pil_img, target_size):
    """PIL image -> float32 array of shape (1, H, W, 3), preprocessed the same
    way for both prediction and Grad-CAM so they always agree."""
    import numpy as np
    img = pil_img.convert("RGB").resize((target_size[1], target_size[0]))
    arr = np.array(img).astype("float32")
    if PREPROCESS_MODE == "scale":
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def run_inference(image_path):
    """Returns (label, confidence, probabilities_dict). Raises RuntimeError
    with a clear message if the model isn't loaded — callers must handle
    this and show it to the user instead of guessing a fake result."""
    if MODEL is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Model is not loaded.")

    import numpy as np
    from PIL import Image

    target_size = MODEL.input_shape[1:3]  # (height, width) from the model itself
    arr = preprocess_pil(Image.open(image_path), target_size)

    preds = MODEL.predict(arr, verbose=0)[0]

    n = min(len(CLASS_NAMES), len(preds))
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(n)}
    idx = int(np.argmax(preds[:n]))
    return CLASS_NAMES[idx], float(preds[idx]), probs


def _jet_colormap(gray):
    """gray: HxW float array in [0,1]. Returns HxWx3 uint8 approximating a
    'jet' colormap (blue -> green -> yellow -> red), without needing
    matplotlib as a dependency."""
    import numpy as np
    r = np.clip(1.5 - np.abs(4 * gray - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * gray - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * gray - 1), 0, 1)
    return np.uint8(255 * np.stack([r, g, b], axis=-1))


def _layer_output_shape(layer):
    """Returns a layer's output shape as a tuple, or None if unavailable.
    Keras 2 exposes layer.output_shape directly; Keras 3 removed that in
    favor of layer.output.shape — this works with either."""
    try:
        shape = layer.output_shape
    except AttributeError:
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(shape, list):
        shape = shape[0] if shape else None
    return shape


def _find_last_conv_layer(model):
    """Finds the last layer with a 4D output (typically the last Conv2D
    layer, or a pretrained base model whose output is a 4D feature map) —
    Grad-CAM needs this. Raises if none exists (e.g. a non-convolutional
    model), which callers must handle."""
    for layer in reversed(model.layers):
        shape = _layer_output_shape(layer)
        if shape is not None and len(shape) == 4:
            return layer
    raise RuntimeError(
        "No convolutional layer found in this model — Grad-CAM requires a "
        "CNN with at least one Conv2D-style layer."
    )


def _pick_gradcam_layer(model):
    """Layer whose feature map Grad-CAM uses. By default: the last 4D layer.
    Set GRADCAM_LAYER=<layer name> in .env to try a different one (an EARLIER
    layer gives a sharper, higher-resolution heatmap)."""
    name = os.environ.get("GRADCAM_LAYER", "").strip()
    if name:
        return model.get_layer(name)
    return _find_last_conv_layer(model)


def _find_nested_base(model):
    """Returns the first top-level layer that is itself a Model (e.g. a
    pretrained EfficientNet/DenseNet base), or None if the model is flat."""
    for layer in model.layers:
        if getattr(layer, "layers", None):
            return layer
    return None


def _gradient_heatmap(arr, class_idx):
    """Classic Grad-CAM. Returns a small 2D heatmap (feature-map resolution).
    arr: tf tensor (1,H,W,3)."""
    import tensorflow as tf

    conv_outputs, grads = None, None
    errors = []

    # Method A: sub-model from the model's own graph (flat models)
    try:
        last_conv_layer = _pick_gradcam_layer(MODEL)
        print(f"[DiaVision] Grad-CAM using layer: {last_conv_layer.name}")
        # Preferred: gradient of the raw pre-softmax score (softmax saturates
        # when the model is confident, which kills the gradient).
        try:
            head = MODEL.layers[-1]
            act_name = getattr(getattr(head, "activation", None), "__name__", "")
            if hasattr(head, "kernel") and act_name in ("softmax", "sigmoid"):
                logit_model = tf.keras.models.Model(
                    inputs=MODEL.inputs,
                    outputs=[last_conv_layer.output, head.input],
                )
                with tf.GradientTape() as tape:
                    conv_outputs, feats = logit_model(arr, training=False)
                    if isinstance(feats, (list, tuple)):
                        feats = feats[0]
                    logits = tf.matmul(feats, head.kernel)
                    if getattr(head, "use_bias", False):
                        logits = logits + head.bias
                    loss = logits[:, class_idx]
                grads = tape.gradient(loss, conv_outputs)
        except Exception as exc:
            errors.append(f"A-logits: {exc}")
            conv_outputs, grads = None, None

        if grads is None:
            grad_model = tf.keras.models.Model(
                inputs=MODEL.inputs,
                outputs=[last_conv_layer.output, MODEL.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(arr, training=False)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                loss = preds[:, class_idx]
            grads = tape.gradient(loss, conv_outputs)
    except Exception as exc:
        errors.append(f"A: {exc}")
        conv_outputs, grads = None, None

    # Method B: nested base model + classifier head
    if grads is None:
        try:
            base = _find_nested_base(MODEL)
            if base is None:
                raise RuntimeError("model has no nested base model")
            inner_conv = _find_last_conv_layer(base)
            inner_model = tf.keras.models.Model(
                inputs=base.inputs,
                outputs=[inner_conv.output, base.output],
            )
            with tf.GradientTape() as tape:
                x = arr
                conv_outputs = None
                for layer in MODEL.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    if layer is base:
                        conv_outputs, x = inner_model(x, training=False)
                    else:
                        x = layer(x)
                preds = x[0] if isinstance(x, (list, tuple)) else x
                loss = preds[:, class_idx]
            grads = tape.gradient(loss, conv_outputs)
        except Exception as exc:
            errors.append(f"B: {exc}")
            conv_outputs, grads = None, None

    if grads is None or conv_outputs is None:
        raise RuntimeError("[v7] Grad-CAM failed. " + " | ".join(errors))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Edge-artifact trim (outer ring of the small feature map)
    trim = int(os.environ.get("GRADCAM_EDGE_TRIM", "1"))
    if trim > 0 and min(heatmap.shape) > 2 * trim + 2:
        heatmap[:trim, :] = 0
        heatmap[-trim:, :] = 0
        heatmap[:, :trim] = 0
        heatmap[:, -trim:] = 0
    return heatmap


def _occlusion_heatmap(arr_np, class_idx, grid=6):
    """Occlusion sensitivity: slide a gray patch over the image and measure how
    much the model's confidence in class_idx drops. Gradient-free, so it is not
    affected by padding/saturation artifacts. Returns an HxW heatmap."""
    import numpy as np

    _, H, W, _ = arr_np.shape
    ph, pw = max(H // 4, 1), max(W // 4, 1)          # patch = 1/4 of the image
    ys = np.linspace(0, H - ph, grid).astype(int)
    xs = np.linspace(0, W - pw, grid).astype(int)
    fill = arr_np.mean(axis=(1, 2), keepdims=True)    # neutral mean-colour patch

    batch = []
    for y in ys:
        for x in xs:
            a = arr_np.copy()
            a[:, y:y + ph, x:x + pw, :] = fill
            batch.append(a[0])
    batch = np.stack(batch)

    base = float(MODEL.predict(arr_np, verbose=0)[0][class_idx])
    probs = MODEL.predict(batch, batch_size=16, verbose=0)[:, class_idx]
    # drop in log-probability: stays sensitive even when the model is ~100% sure
    drops = np.maximum(np.log(base + 1e-7) - np.log(probs + 1e-7), 0)

    heat = np.zeros((H, W), dtype="float32")
    cnt = np.zeros((H, W), dtype="float32")
    i = 0
    for y in ys:
        for x in xs:
            heat[y:y + ph, x:x + pw] += drops[i]
            cnt[y:y + ph, x:x + pw] += 1
            i += 1
    heat = heat / np.maximum(cnt, 1)
    return heat


def generate_gradcam(image_path, class_idx):
    """Builds an explanation heatmap overlay for the given image and predicted
    class index, saves it next to the original upload, and returns the saved
    filename. Raises RuntimeError with a clear message on failure (callers
    catch it and show it on the result page).

    Method is chosen by GRADCAM_METHOD in .env:
        gradcam (default)   - classic Grad-CAM, fast
        occlusion           - gradient-free, slower; use if Grad-CAM shows edge artifacts"""
    if MODEL is None:
        raise RuntimeError("Model is not loaded.")

    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageFilter

    method = os.environ.get("GRADCAM_METHOD", "gradcam").strip().lower()
    target_size = MODEL.input_shape[1:3]
    original = Image.open(image_path).convert("RGB")
    arr_np = preprocess_pil(original, target_size)

    try:
        if method == "gradcam":
            heatmap = _gradient_heatmap(tf.convert_to_tensor(arr_np), class_idx)
        else:
            heatmap = _occlusion_heatmap(arr_np, class_idx)
    except Exception as exc:
        raise RuntimeError(f"[v7:{method}] {exc}")

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original.size, resample=Image.BICUBIC)
    heat_img = heat_img.filter(ImageFilter.GaussianBlur(3))
    heat = np.array(heat_img).astype("float32") / 255.0

    # Keep the heatmap on the subject: mask out pure-black background.
    gray = np.array(original).astype("float32").mean(axis=2)
    mask_img = Image.fromarray(np.uint8((gray > 20) * 255)).filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(3))
    mask = np.array(mask_img).astype("float32") / 255.0
    heat = heat * mask
    if heat.max() > 0:
        heat = heat / heat.max()

    colored = _jet_colormap(heat)
    original_arr = np.array(original).astype("float32")
    alpha = (0.65 * heat)[..., None]
    overlay = np.clip(original_arr * (1 - alpha) + colored.astype("float32") * alpha, 0, 255).astype("uint8")

    gradcam_filename = f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    Image.fromarray(overlay).save(os.path.join(UPLOAD_FOLDER, gradcam_filename))
    return gradcam_filename


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=bool(request.form.get("remember")))
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "").strip() or "Clinician"
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill in all required fields.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
        else:
            user = User(name=name, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Account created — welcome to DiaVision AI.", "success")
            return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Core pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    total_patients = Patient.query.count()
    total_analyses = Analysis.query.count()
    recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(5).all()
    recent_patients = Patient.query.order_by(Patient.created_at.desc()).limit(5).all()
    flagged = Analysis.query.filter(Analysis.prediction_label != "No DR").count()
    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_analyses=total_analyses,
        flagged=flagged,
        recent_analyses=recent_analyses,
        recent_patients=recent_patients,
        model_loaded=(MODEL is not None),
        model_error=MODEL_LOAD_ERROR,
    )


@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    patients = Patient.query.order_by(Patient.name).all()

    if request.method == "POST":
        if MODEL is None:
            flash(f"Cannot run analysis — model not loaded: {MODEL_LOAD_ERROR}", "error")
            return redirect(url_for("analysis"))

        patient_id = request.form.get("patient_id")

        if patient_id == "new":
            code = request.form.get("patient_code") or f"PX-{int(datetime.utcnow().timestamp())}"
            if Patient.query.filter_by(patient_code=code).first():
                flash("That Patient ID already exists.", "error")
                return redirect(url_for("analysis"))
            patient = Patient(
                patient_code=code,
                name=request.form.get("name", "Unnamed Patient"),
                age=request.form.get("age") or None,
                gender=request.form.get("gender"),
                diabetes_type=request.form.get("diabetes_type"),
                duration_years=request.form.get("duration_years") or None,
                city=request.form.get("city"),
                notes=request.form.get("notes"),
            )
            db.session.add(patient)
            db.session.commit()
        else:
            patient = Patient.query.get(int(patient_id))
            if not patient:
                flash("Please select a valid patient.", "error")
                return redirect(url_for("analysis"))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please attach a retinal image to analyze.", "error")
            return redirect(url_for("analysis"))
        if not allowed_file(file.filename):
            flash("Unsupported file type.", "error")
            return redirect(url_for("analysis"))

        filename = secure_filename(f"{patient.patient_code}_{int(datetime.utcnow().timestamp())}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            label, confidence, probs = run_inference(save_path)
        except Exception as exc:
            flash(f"Analysis failed: {exc}", "error")
            return redirect(url_for("analysis"))

        # Grad-CAM is best-effort: if it fails for any reason (unsupported
        # architecture, gradient issue, etc.) we still keep the prediction
        # and show a clear technical-issue notice on the result page
        # instead of losing the analysis or crashing.
        gradcam_filename = None
        gradcam_error = None
        try:
            class_idx = CLASS_NAMES.index(label)
            gradcam_filename = generate_gradcam(save_path, class_idx)
        except Exception as exc:
            gradcam_error = str(exc)
            app.logger.warning("Grad-CAM generation failed: %s", exc)

        record = Analysis(
            patient_id=patient.id,
            image_filename=filename,
            prediction_label=label,
            confidence=confidence,
            probabilities_json=json.dumps(probs),
            reviewed_by=current_user.name,
            gradcam_filename=gradcam_filename,
            gradcam_error=gradcam_error,
        )
        db.session.add(record)
        db.session.commit()

        return redirect(url_for("result", analysis_id=record.id))

    return render_template("analysis.html", patients=patients, model_loaded=(MODEL is not None), model_error=MODEL_LOAD_ERROR)


@app.route("/result/<int:analysis_id>")
@login_required
def result(analysis_id):
    record = Analysis.query.get_or_404(analysis_id)
    sorted_probs = sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return render_template("result.html", record=record, patient=record.patient, sorted_probs=sorted_probs)


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
@app.route("/patients")
@login_required
def patients():
    q = request.args.get("q", "").strip()
    query = Patient.query
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(Patient.name.ilike(like), Patient.patient_code.ilike(like)))
    all_patients = query.order_by(Patient.created_at.desc()).all()
    return render_template("patients.html", patients=all_patients, q=q)


@app.route("/patients/<int:patient_id>")
@login_required
def patient_profile(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template("patient_profile.html", patient=patient)


# ---------------------------------------------------------------------------
# History / Reports / Analytics
# ---------------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    filter_label = request.args.get("label", "")
    query = Analysis.query
    if filter_label:
        query = query.filter_by(prediction_label=filter_label)
    records = query.order_by(Analysis.created_at.desc()).all()
    return render_template("history.html", records=records, class_names=CLASS_NAMES, filter_label=filter_label)


@app.route("/reports")
@login_required
def reports():
    records = Analysis.query.order_by(Analysis.created_at.desc()).all()
    return render_template("reports.html", records=records)


@app.route("/reports/<int:analysis_id>/download")
@login_required
def download_report(analysis_id):
    record = Analysis.query.get_or_404(analysis_id)
    patient = record.patient

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    filename = f"report_{record.id}_{patient.patient_code}.pdf"
    filepath = os.path.join(app.config["REPORTS_FOLDER"], filename)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBrand", parent=styles["Title"], textColor=colors.HexColor("#5b21b6"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#333333"))
    body_style = styles["BodyText"]
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["BodyText"],
                                       textColor=colors.HexColor("#8a1f11"), fontSize=8, leading=11)

    doc = SimpleDocTemplate(filepath, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = [
        Paragraph("DiaVision AI — Screening Report", title_style),
        Spacer(1, 4),
        HRFlowable(width="100%", color=colors.HexColor("#8B5CF6"), thickness=1.2),
        Spacer(1, 14),
        Paragraph("Patient Information", h2_style),
        Table([
            ["Name", patient.name, "Patient ID", patient.patient_code],
            ["Age", str(patient.age or "N/A"), "Gender", patient.gender or "N/A"],
            ["Diabetes Type", patient.diabetes_type or "N/A", "City", patient.city or "N/A"],
        ], colWidths=[90, 150, 90, 130], style=TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#555555")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])),
        Spacer(1, 16),
        Paragraph("AI Screening Result", h2_style),
        Paragraph(f"<b>Primary Finding:</b> {record.prediction_label}", body_style),
        Paragraph(f"<b>Confidence:</b> {record.confidence * 100:.1f}%", body_style),
        Paragraph(f"<b>Screening Date:</b> {record.created_at.strftime('%Y-%m-%d %H:%M UTC')}", body_style),
        Paragraph(f"<b>Reviewed By:</b> {record.reviewed_by or 'Pending review'}", body_style),
        Spacer(1, 10),
        Paragraph("Class Probabilities", h2_style),
    ]

    prob_rows = [["Finding", "Probability"]]
    for label, prob in sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True):
        prob_rows.append([label, f"{prob * 100:.1f}%"])
    story.append(Table(prob_rows, colWidths=[300, 120], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
    ])))

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], record.image_filename)
    if os.path.exists(image_path):
        story.append(Spacer(1, 16))
        story.append(Paragraph("Submitted Image", h2_style))
        try:
            story.append(RLImage(image_path, width=220, height=220, kind="proportional"))
        except Exception:
            pass

    # Include the Grad-CAM overlay in the report when it exists.
    if record.gradcam_filename:
        gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], record.gradcam_filename)
        if os.path.exists(gradcam_path):
            story.append(Spacer(1, 16))
            story.append(Paragraph("Grad-CAM Heatmap", h2_style))
            try:
                story.append(RLImage(gradcam_path, width=220, height=220, kind="proportional"))
            except Exception:
                pass

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=0.7))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This AI-generated analysis is intended for clinical decision support only and does not "
        "constitute a final medical diagnosis. All results must be reviewed by a board-certified "
        "healthcare professional before any treatment planning.", disclaimer_style))

    doc.build(story)
    return send_from_directory(app.config["REPORTS_FOLDER"], filename, as_attachment=True)


@app.route("/analytics")
@login_required
def analytics():
    counts = {name: 0 for name in CLASS_NAMES}
    for label, in db.session.query(Analysis.prediction_label).all():
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values()) or 1
    avg_confidence = db.session.query(db.func.avg(Analysis.confidence)).scalar() or 0
    return render_template(
        "analytics.html",
        counts=counts,
        total=total,
        avg_confidence=avg_confidence,
        total_patients=Patient.query.count(),
        total_analyses=Analysis.query.count(),
    )


# ---------------------------------------------------------------------------
# Assistant — supports either Grok (xAI, api.x.ai) or Groq Cloud
# (api.groq.com — fast hosted inference for models like Llama). These are
# two different services with different keys, easy to mix up by name.
# Set GROK_API_KEY for xAI's Grok, or GROQ_API_KEY for Groq Cloud — set
# either (or both). Falls back to a rule-based responder if neither key
# works or the call fails for any reason.
# ---------------------------------------------------------------------------
GROK_API_KEY = os.environ.get("GROK_API_KEY", "").strip()
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if GROK_API_KEY:
    print(f"[DiaVision] Grok (xAI) assistant: ENABLED ({GROK_MODEL})")
if GROQ_API_KEY:
    print(f"[DiaVision] Groq Cloud assistant: ENABLED ({GROQ_MODEL})")
if not GROK_API_KEY and not GROQ_API_KEY:
    print("[DiaVision] No GROK_API_KEY or GROQ_API_KEY set — chatbot using rule-based fallback only.")

ASSISTANT_SYSTEM_PROMPT = (
    "You are the DiaVision Assistant, embedded in a diabetic retinopathy "
    "screening dashboard used by clinicians. Be concise and clinically "
    "accurate. You can explain DR grading, general diabetic eye disease "
    "education, suggest sensible next clinical steps for a given finding "
    "(e.g. follow-up interval, referral to ophthalmology), and answer "
    "questions about using the app (New Analysis, Patients, History, "
    "Reports, Analytics). You do not have live access to this session's "
    "actual patient records or images unless the user pastes details into "
    "the chat. Never present your output as a final diagnosis or treatment "
    "order — always note that AI suggestions require review by the "
    "attending clinician before any decision is acted on."
)

KB = [
    (["dr", "retinopathy", "what is"], "Diabetic Retinopathy (DR) is damage to the retina's blood "
     "vessels caused by prolonged high blood sugar. It's graded from No DR to Proliferative DR "
     "based on severity of microaneurysms, hemorrhages, and neovascularization."),
    (["confidence", "accuracy", "how accurate"], "Model confidence reflects the predicted probability "
     "for the top class. Always correlate high-confidence AI findings with clinical exam before "
     "treatment decisions — this tool supports, not replaces, a clinician's judgment."),
    (["upload", "new analysis", "how do i"], "Go to New Analysis, pick or add a patient, then upload "
     "a retinal image. You'll be taken straight to the result page once processing finishes."),
    (["report", "download", "pdf"], "You can download a PDF screening report from the Reports "
     "page or directly from any result page."),
]


def rule_based_reply(message):
    text = message.lower()
    for keywords, answer in KB:
        if any(k in text for k in keywords):
            return answer
    return ("I can help with questions about diabetic retinopathy grading, using the analysis "
            "workflow, or your reports. Could you rephrase your question? "
            "(Set GROK_API_KEY or GROQ_API_KEY in .env for richer, LLM-powered answers.)")


def _call_openai_compatible(url, api_key, model, message, history):
    import requests
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.4},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def grok_reply(message, history):
    """Calls xAI's Grok API. Returns None (never raises) on any failure so
    the caller can fall back cleanly — missing key, network error, bad
    response, whatever."""
    if not GROK_API_KEY:
        return None
    try:
        return _call_openai_compatible(GROK_API_URL, GROK_API_KEY, GROK_MODEL, message, history)
    except Exception as exc:  # pragma: no cover
        app.logger.warning("Grok (xAI) API call failed, falling back: %s", exc)
        return None


def groq_reply(message, history):
    """Calls Groq Cloud's OpenAI-compatible API. Same never-raises contract
    as grok_reply()."""
    if not GROQ_API_KEY:
        return None
    try:
        return _call_openai_compatible(GROQ_API_URL, GROQ_API_KEY, GROQ_MODEL, message, history)
    except Exception as exc:  # pragma: no cover
        app.logger.warning("Groq Cloud API call failed, falling back: %s", exc)
        return None


def llm_reply(message, history):
    """Tries Groq first (since that's what most people setting GROQ_API_KEY
    actually want — fast + often free-tier), then Grok, then the
    rule-based fallback. Returns (reply, source)."""
    for name, fn in (("groq", groq_reply), ("grok", grok_reply)):
        reply = fn(message, history)
        if reply is not None:
            return reply, name
    return rule_based_reply(message), "rules"


@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", grok_enabled=bool(GROK_API_KEY), groq_enabled=bool(GROQ_API_KEY))


@app.route("/api/chatbot", methods=["POST"])
@login_required
def api_chatbot():
    from flask import session

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a question."}), 400

    history = session.get("chat_history", [])
    reply, source = llm_reply(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    session["chat_history"] = history[-12:]  # keep last ~6 exchanges for context

    return jsonify({"reply": reply, "source": source})


@app.route("/api/chatbot/reset", methods=["POST"])
@login_required
def api_chatbot_reset():
    from flask import session
    session.pop("chat_history", None)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------
@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def ensure_schema():
    """Lightweight auto-migration: adds any columns this version of the
    app expects that an existing diavision.db (or Turso DB) doesn't have
    yet, so upgrading the app doesn't require deleting your database.
    Safe to run every startup — it's a no-op once columns already exist."""
    try:
        with db.engine.connect() as conn:
            existing = {row[1] for row in conn.execute(db.text("PRAGMA table_info(analysis)")).fetchall()}
            statements = []
            if "gradcam_filename" not in existing:
                statements.append("ALTER TABLE analysis ADD COLUMN gradcam_filename VARCHAR(255)")
            if "gradcam_error" not in existing:
                statements.append("ALTER TABLE analysis ADD COLUMN gradcam_error TEXT")
            for stmt in statements:
                conn.execute(db.text(stmt))
            if statements:
                conn.commit()
                print(f"[DiaVision] Migrated database schema ({len(statements)} column(s) added).")
    except Exception as exc:  # pragma: no cover
        print(f"[DiaVision] Schema check skipped (non-fatal): {exc}")


def seed_demo_data():
    db.create_all()
    ensure_schema()
    if not User.query.filter_by(email="doctor@diavision.ai").first():
        demo = User(name="Dr. Julian Vance", email="doctor@diavision.ai", role="Senior Endocrinologist")
        demo.set_password("password123")
        db.session.add(demo)
        db.session.commit()


if __name__ == "__main__":
    with app.app_context():
        seed_demo_data()
    app.run(debug=False, use_reloader=False)
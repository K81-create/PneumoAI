

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import numpy as np
from gradcam import generate_gradcam
import os


app = Flask(__name__)
app.secret_key = "pneumoai_secret_key_2026"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumoai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class User(db.Model):
    """User account model with hashed password storage."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Stores each prediction result linked to a user."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(20), nullable=False)       # PNEUMONIA or NORMAL
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.result} – {self.confidence}%>'


# ---------------------------------------------------------------------------
# Load Trained CNN Model
# ---------------------------------------------------------------------------
model = load_model("models/pneumonia_model.h5")


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes – Public Pages
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    """Landing page with hero section and project overview."""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with password hashing."""
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('This username is already taken.', 'danger')
            return redirect(url_for('register'))

        # Create new user with hashed password
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with session management."""
    if request.method == 'POST':
        email = request.form['email'].strip()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            # Store user info in session
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid email or password.', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clear session and redirect to home."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# ---------------------------------------------------------------------------
# Routes – Authenticated Pages
# ---------------------------------------------------------------------------

@app.route('/dashboard')
def dashboard():
    """User dashboard with upload form. Requires login."""
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    # Get count of total predictions for this user
    prediction_count = Prediction.query.filter_by(user_id=user.id).count()

    return render_template('dashboard.html', user=user, prediction_count=prediction_count)


@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded X-ray image and return prediction."""
    if 'user_id' not in session:
        flash('Please log in to make predictions.', 'warning')
        return redirect(url_for('login'))

    # Validate file upload
    if 'file' not in request.files:
        flash('No file uploaded.', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'danger')
        return redirect(url_for('dashboard'))

    # Save the uploaded file
    filename = secure_filename(file.filename)
    # Add timestamp to avoid filename collisions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image for CNN model
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Run prediction
    prediction_value = model.predict(img_array)

    # Determine result and confidence
    if prediction_value[0][0] > 0.5:
        result = "PNEUMONIA"
        confidence = round(float(prediction_value[0][0]) * 100, 2)
    else:
        result = "NORMAL"
        confidence = round((1 - float(prediction_value[0][0])) * 100, 2)

    # Generate Grad-CAM heatmap overlay
    gradcam_filename = "gradcam_" + filename
    gradcam_save_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)

    gradcam_result = generate_gradcam(
        model,
        img_array,
        filepath,
        gradcam_save_path
    )

    # Build the Grad-CAM image path for the template (None if generation failed)
    gradcam_image_path = None
    if gradcam_result is not None:
        gradcam_image_path = gradcam_save_path.replace('\\', '/')

    # Save prediction to database
    new_prediction = Prediction(
        user_id=session['user_id'],
        image_name=filename,
        result=result,
        confidence=confidence,
        created_at=datetime.utcnow()
    )
    db.session.add(new_prediction)
    db.session.commit()

    # Get user data for the dashboard
    user = User.query.get(session['user_id'])
    prediction_count = Prediction.query.filter_by(user_id=user.id).count()

    return render_template(
        'dashboard.html',
        user=user,
        prediction_count=prediction_count,
        prediction=result,
        confidence=confidence,
        image_path=filepath.replace('\\', '/'),
        gradcam_image_path=gradcam_image_path
    )


@app.route('/history')
def history():
    """Show prediction history for the logged-in user."""
    if 'user_id' not in session:
        flash('Please log in to view prediction history.', 'warning')
        return redirect(url_for('login'))

    # Get all predictions for this user, newest first
    predictions = Prediction.query.filter_by(
        user_id=session['user_id']
    ).order_by(Prediction.created_at.desc()).all()

    return render_template('history.html', predictions=predictions)


# Create database tables when the app starts
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
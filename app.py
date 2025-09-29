from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session
import pandas as pd
import sqlite3
import os
import csv
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app import app as application

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Needed for sessions

# Constants
USERS_CSV = 'users.csv'
ADMIN_EMAIL = 'admin@gmail.com'
ADMIN_PASSWORD = 'admin'
DB_FILE = "users.db"

# Load dataset
df = pd.read_csv("Complete_dataset.csv", on_bad_lines="skip", engine="python")
df.dropna(subset=['Percentile', 'Category', 'Branch', 'College Name'], inplace=True)

# Label Encoding
le_category = LabelEncoder()
le_branch = LabelEncoder()
le_college = LabelEncoder()
df['Category_encoded'] = le_category.fit_transform(df['Category'])
df['Branch_encoded'] = le_branch.fit_transform(df['Branch'])
df['College_encoded'] = le_college.fit_transform(df['College Name'])

X = df[['Percentile', 'Category_encoded', 'Branch_encoded']]
y = df['College_encoded']

# Train RandomForest model
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X, y)

CATEGORY_MAP = {
    "OPEN": [
        "GOPENH","GOPENO","GOPENS",
        "LOPENH","LOPENO","LOPENS"
    ],
    "OBC": [
        "DEFOBCS","DEFROBCS","GOBCS","LOBCS",
        "GOBCH","GOBCO","LOBCO","LOBCH"
    ],
    "SC/ST": [
        "DEFRSCS","DEFSCS",
        "GSCS","GSCH","GSCO",
        "LSTS","LSTH","LSTO",
        "LVJS","LVJH","LVJO"
    ],
    "TFWS": ["TFWS"],
    "EWS": ["EWS"],
    "PWD": [
        "PWDOBCH","PWDOBCS","PWDOPENH","PWDOPENS",
        "PWDRNT1S","PWDRNT2S","PWDRNT3S",
        "PWDROBCH","PWDROBCS","PWDRSCH","PWDRSCS",
        "PWDRVJS","PWDSCH","PWDSCS"
    ],
    "MI": ["MI","MI-AI","MI-MH"],
    "ORPHAN": ["ORPHAN"],
    "GNT": [
        "GNT1H","GNT1O","GNT1S",
        "GNT2H","GNT2O","GNT2S",
        "GNT3H","GNT3O","GNT3S"
    ],
    "LNT": [
        "LNT1H","LNT1O","LNT1S",
        "LNT2H","LNT2O","LNT2S",
        "LNT3H","LNT3O","LNT3S"
    ]
}

# ========== Generalized Branch Map ==========
BRANCH_MAP = {
    "CSE (AI/ML)": [
        "Computer Science and Engineering(Artificial Intelligence and Machine Learning)",
        "Computer Science and Engineering (Artificial Intelligence)",
        "Artificial Intelligence (AI) and Data Science",
        "Artificial Intelligence and Data Science",
        "Artificial Intelligence and Machine Learning",
        "Computer Science and Engineering (Artificial Intelligence and Data Science)",
        "Artificial Intelligence and Data Science University  Jalgaon",
        "Computer Science and Engineering University  Jalgaon",
        "Artificial Intelligence"
    ],
    "CSE (Data Science)": [
        "Computer Science and Engineering(Data Science)",
        "Data Science",
        "Data Engineering"
    ],
    "CSE (Cyber/IoT)": [
        "Computer Science and Engineering (Cyber Security)",
        "Computer Science and Engineering (Internet of Things and Cyber Security Including Block Chain Technology)",
        "Computer Science and Engineering (IoT)",
        "Internet of Things (IoT)",
        "Industrial IoT",
        "Cyber Security"
    ],
    "CSE (General)": [
        "Computer Science and Engineering",
        "Computer Engineering",
        "Computer Science and Information Technology",
        "Computer Science and Technology",
        "Computer Science and Design",
        "Computer Science and Business Systems",
        "Computer Technology",
        "Information Technology"
    ],
    "Electronics and Telecommunication Engineering": [
        "Electronics and Telecommunication Engg",
        "Electronics and Communication Engineering",
        "Electronics Engineering",
        "Electronics and Computer Science",
        "Electronics and Computer Engineering",
        "Electronics Engineering ( VLSI Design and Technology)",
        "Electrical and Electronics Engineering",
        "Electronics and Telecommunication Engg University Jalgaon"
    ],
    "Electrical Engineering": [
        "Electrical Engineering",
        "Electrical Engg[Electronics and Power]",
        "Electrical Engg [Electrical and Power]",
        "Electrical and ComputerEngineering"
    ],
    "Mechanical Engineering": [
        "Mechanical Engineering",
        "Mechanical Engineering[Sandwich]",
        "Mechanical & Automation Engineering",
        "Mechanical and Mechatronics Engineering (Additive Manufacturing)",
        "Production Engineering",
        "Production Engineering[Sandwich]",
        "Manufacturing Science and Engineering"
    ],
    "Robotics / Automation": [
        "Robotics and Automation",
        "Automation and Robotics",
        "Robotics",
        "Robotics and Artificial Intelligence"
    ],
    "Civil Engineering": [
        "Civil Engineering",
        "Civil and Environmental Engineering",
        "Civil and infrastructure Engineering",
        "Structural Engineering"
    ],
    "Chemical Engineering": [
        "Chemical Engineering"
    ],
    "Petro / Oil / Surface Coating": [
        "Petro Chemical Engineering",
        "Petro Chemical Technology",
        "Oil Technology",
        "OilOleochemicals and Surfactants Technology",
        "Surface Coating Technology",
        "Oil and Paints Technology",
        "Paints Technology",
        "Dyestuff Technology"
    ],
    "Pharmaceutical / Fine Chemicals": [
        "Pharmaceuticals Chemistry and Technology",
        "Pharmaceutical and Fine Chemical Technology"
    ],
    "Textile / Polymer": [
        "Textile Technology",
        "Textile Plant Engineering",
        "Textile Engineering / Technology",
        "Textile Chemistry",
        "Fibres and Textile Processing Technology",
        "Man Made Textile Technology",
        "Polymer Engineering and Technology",
        "Plastic and Polymer Technology",
        "Plastic and Polymer Engineering",
        "Plastic Technology",
        "Paper and Pulp Technology"
    ],
    "Food / Agriculture": [
        "Food Technology",
        "Food Technology And Management",
        "Food Engineering and Technology",
        "Agricultural Engineering",
        "Agriculture Engineering"
    ],
    "Fashion Technology": [
        "Fashion Technology"
    ],
    "Instrumentation / Controls": [
        "Instrumentation and Control Engineering",
        "Instrumentation Engineering"
    ],
    "Mechatronics Engineering": [
        "Mechatronics Engineering"
    ],
    "Aeronautical Engineering": [
        "Aeronautical Engineering"
    ],
    "Automobile / Automotive": [
        "Automobile Engineering",
        "Automotive Technology"
    ],
    "Mining Engineering": [
        "Mining Engineering"
    ],
    "Safety / Fire Engineering": [
        "Safety and Fire Engineering"
    ],
    "Metallurgy / Material Technology": [
        "Metallurgy and Material Technology"
    ],
    "Bio-Medical / Bio-Technology":[
        "Bio Medical Engineering",
        "Bio Technology"
    ],
    "Printing Technology":[
        "Printing Technology"
    ]
}




# Load college links
with open('static/js/College_list_links.json', 'r', encoding='utf-8') as f:
    college_links = json.load(f)

# ================== Database Functions ==================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            gender TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def export_to_csv():
    conn = sqlite3.connect(DB_FILE)
    df_users = pd.read_sql_query("SELECT * FROM users", conn)
    df_users.to_csv(USERS_CSV, index=False)
    conn.close()

def read_users():
    users = []
    if os.path.exists(USERS_CSV):
        with open(USERS_CSV, newline='') as f:
            reader = csv.DictReader(f)
            users = list(reader)
    return users



# ================== Admin Metrics ==================
def read_users_from_db():
    """
    Read all users from the SQLite database.
    Returns a list of dicts with keys: id, name, email, phone, gender
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # to get dict-like rows
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    rows = c.fetchall()
    users = [dict(row) for row in rows]
    conn.close()
    return users

def get_metrics():
    """
    Returns total, male, female, and other counts from database users.
    """
    users = read_users_from_db()
    total = len(users)
    male = len([u for u in users if str(u.get('gender','')).strip().lower() == 'male'])
    female = len([u for u in users if str(u.get('gender','')).strip().lower() == 'female'])
    other = total - male - female
    return {"total": total, "male": male, "female": female, "other": other}

# Initialize database
init_db()

# ================== Routes ==================
@app.route('/')
def start():
    return render_template('start.html')

@app.route('/form', methods=['GET', 'POST'])
def form_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        gender = request.form.get('gender')

        if not (name and email and phone and gender):
            return render_template('form.html', error="Please fill all fields")

        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (name, email, phone, gender) VALUES (?, ?, ?, ?)",
                (name, email, phone, gender)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('form.html', error="Email already registered")
        conn.close()

        export_to_csv()  # update CSV after new user
        return redirect(url_for('predictor_page'))

    return render_template('form.html')

# ================== Login & Logout ==================
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Validation
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['email'] = email
            session['is_admin'] = True
            return redirect(url_for('admin_index'))  # Redirect to admin dashboard
        else:
            return render_template('admin_login.html', error="Invalid credentials")

    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('form_page'))

@app.route('/admin_index')
def admin_index():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    return render_template('admin_index.html')

# ================== Recommender Pages ==================
@app.route('/predictor')
def predictor_page():
    return render_template('index.html')

@app.route('/predictor_location')
def predictor_location_page():
    return render_template('location_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        percentile = float(data.get('percentile', 0))
        category_key = data.get('category', "").strip().upper()
        branch_key = data.get('branch', "").strip()

        # Map frontend category to CSV subcategories
        categories_to_check = CATEGORY_MAP.get(category_key, [category_key])

        # Map frontend branch to CSV branches
        branches_to_check = BRANCH_MAP.get(branch_key, [branch_key])

        # Ensure CSV columns are clean
        df['Category'] = df['Category'].str.strip()
        df['Branch'] = df['Branch'].str.strip()
        df['Percentile'] = pd.to_numeric(df['Percentile'], errors='coerce')

        # Filter dataset
        filtered = df[
            (df['Percentile'] <= percentile) &
            (df['Category'].isin(categories_to_check)) &
            (df['Branch'].isin(branches_to_check))
        ]

        # Sort by percentile descending and get top 10
        top_colleges = filtered.sort_values(by='Percentile', ascending=False).head(10)

        # Add Link column if missing
        if 'Link' not in top_colleges.columns:
            top_colleges['Link'] = "#"

        results = top_colleges[['College Name', 'Branch', 'Category', 'Percentile', 'Link']].to_dict(orient='records')
        return jsonify(results)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Server error occurred.'}), 500

    
@app.route('/predict_location', methods=['POST'])
def predict_with_location():
    try:
        data = request.get_json()
        percentile = float(data.get('percentile', 0))
        category_key = data.get('category', "").strip().upper()
        branch_key = data.get('branch', "").strip()
        location_keyword = data.get('location', "").strip().lower()

        # Map frontend category and branch
        categories_to_check = CATEGORY_MAP.get(category_key, [category_key])
        branches_to_check = BRANCH_MAP.get(branch_key, [branch_key])

        # Filter dataset
        df['Category'] = df['Category'].str.strip()
        df['Branch'] = df['Branch'].str.strip()
        df['College Name'] = df['College Name'].str.strip()
        df['Percentile'] = pd.to_numeric(df['Percentile'], errors='coerce')

        filtered = df[
            (df['Percentile'] <= percentile) &
            (df['Category'].isin(categories_to_check)) &
            (df['Branch'].isin(branches_to_check)) &
            (df['College Name'].str.lower().str.contains(location_keyword))
        ]

        # Sort by percentile descending and get top 10
        top_colleges = filtered.sort_values(by='Percentile', ascending=False).head(10)

        # Add Link column from college_links if available
        results = []
        for _, row in top_colleges.iterrows():
            college_name = row['College Name']
            link = college_links.get(college_name, "#")
            results.append({
                "College Name": college_name,
                "Branch": row['Branch'],
                "Category": row['Category'],
                "Percentile": row['Percentile'],
                "Link": link
            })

        return jsonify(results)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Server error occurred.'}), 500

# ================== College Lookup ==================
@app.route('/college_lookup')
def college_lookup():
    return render_template('college_lookup.html')

@app.route('/lookup_college_data', methods=['GET'])
def lookup_college_data():
    query = request.args.get('query', '').strip().lower()
    category = request.args.get('category', '').strip().upper()
    if not query:
        return jsonify([])

    df_lookup = pd.read_csv("Complete_dataset.csv", on_bad_lines='skip', engine='python')
    df_lookup.dropna(subset=['College Name', 'Branch', 'Category', 'Percentile'], inplace=True)
    filtered = df_lookup[df_lookup['College Name'].str.lower().str.contains(query)]
    if category:
        filtered = filtered[filtered['Category'].str.upper() == category]
    filtered = filtered.sort_values(by='Percentile', ascending=False)
    result = filtered[['College Name', 'Branch', 'Category', 'Percentile']].to_dict(orient='records')
    return jsonify(result)


@app.route('/metrics')
def metrics_page():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    return render_template('admin_metrics.html')

@app.route('/get_user_metrics')
def get_user_metrics():
    return jsonify(get_metrics())

@app.route('/download_users_csv')
def download_users_csv():
    """
    Export all users from the database to a CSV and send as a download.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        df_users = pd.read_sql_query("SELECT * FROM users", conn)
        conn.close()

        if df_users.empty:
            return "No data available", 404

        # Save to a temporary CSV file
        temp_csv = "users_export.csv"
        df_users.to_csv(temp_csv, index=False)
        return send_file(temp_csv, as_attachment=True)

    except Exception as e:
        return f"Error exporting users: {str(e)}", 500


# ================== API to register user (optional) ==================
@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.json
    new_user = {
        'name': data['name'],
        'email': data['email'],
        'phone': data['phone'],
        'gender': data['gender']
    }
    file_exists = os.path.exists(USERS_CSV)
    with open(USERS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','email','phone','gender'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_user)
    return jsonify({"success": True})

# ================== Run Flask ==================
if __name__ == '__main__':
    app.run(debug=True)

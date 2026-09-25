import sqlite3
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            attendance REAL NOT NULL,
            marks REAL NOT NULL,
            status TEXT NOT NULL,
            risk_level TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- ML MODEL SETUP ---
# Feature 1: Attendance Percentage | Feature 2: Marks
# Target: 1 = At Risk, 0 = Safe
X_train = np.array([
    [90, 85], [80, 75], [95, 90], [85, 80],  # Safe students
    [40, 45], [50, 30], [30, 60], [55, 40],  # At Risk students
    [70, 50], [60, 65]                       # Borderline cases
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

model = LogisticRegression()
model.fit(X_train, y_train)

def predict_risk(attendance, marks):
    prediction = model.predict([[attendance, marks]])[0]
    return "At Risk" if prediction == 1 else "Safe"

# --- ROUTES & API ENDPOINTS ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students')
    rows = cursor.fetchall()
    conn.close()

    students = []
    for row in rows:
        students.append({
            'id': row[0],
            'name': row[1],
            'attendance': row[2],
            'marks': row[3],
            'status': row[4],
            'risk_level': row[5]
        })
    
    avg_marks = np.mean([s['marks'] for s in students]) if students else 0
    
    return jsonify({
        'students': students,
        'class_average': round(float(avg_marks), 2)
    })

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.get_json()
    name = data.get('name')
    attendance = float(data.get('attendance'))
    marks = float(data.get('marks'))

    status = "Pass" if marks >= 40 else "Fail"
    risk_level = predict_risk(attendance, marks)

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO students (name, attendance, marks, status, risk_level)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, attendance, marks, status, risk_level))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Student added successfully!'}), 201

# --- BACKEND DELETE / UNDO ROUTE ---
@app.route('/api/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students WHERE id = ?', (student_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Record deleted successfully!'}), 200

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
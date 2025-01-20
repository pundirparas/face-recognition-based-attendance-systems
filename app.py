from flask import Flask, request, render_template, redirect, url_for
import os
from datetime import date, datetime
import pandas as pd
import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import traceback
import shutil

# Initialize Flask App
app = Flask(__name__)

# Directory Setup
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Variables for Date
def get_current_date():
    return date.today().strftime("%m_%d_%y")

def get_attendance_file():
    current_date = get_current_date()
    return f'Attendance/Attendance-{current_date}.csv'

# Ensure Today's Attendance File Exists
def create_attendance_file():
    attendance_file = get_attendance_file()
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time\n')

# Helper Functions
def total_registered_users():
    try:
        return len([name for name in os.listdir('static/faces') if os.path.isdir(os.path.join('static/faces', name))])
    except Exception as e:
        print(f"Error counting registered users: {e}")
        return 0

def extract_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def train_model():
    faces, labels = [], []
    try:
        for user in os.listdir('static/faces'):
            user_path = os.path.join('static/faces', user)
            if os.path.isdir(user_path):
                for img_file in os.listdir(user_path):
                    img_path = os.path.join(user_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        resized_face = cv2.resize(img, (50, 50)).flatten()
                        faces.append(resized_face)
                        labels.append(user)
        
        if faces and labels:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(faces, labels)
            joblib.dump(knn, 'static/face_recognition_model.pkl')
            return True
        return False
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def add_attendance(user_id):
    try:
        # Extract name and roll number from the user_id
        name, roll = user_id.split('_')
        current_time = datetime.now().strftime("%H:%M:%S")
        attendance_file = get_attendance_file()
        
        # Read the current attendance data
        df = pd.read_csv(attendance_file) if os.path.exists(attendance_file) else pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        # Check if the roll number already exists in today's attendance
        if roll not in df['Roll'].astype(str).tolist():
            # If not, create a new entry
            new_entry = pd.DataFrame({'Name': [name], 'Roll': [roll], 'Time': [current_time]})
            df = pd.concat([df, new_entry], ignore_index=True)
        
        # Drop duplicates in case the user is detected more than once on the same day
        df.drop_duplicates(subset='Roll', keep='first', inplace=True)
        
        # Save the updated attendance data to the file
        df.to_csv(attendance_file, index=False)
        
    except Exception as e:
        print(f"Error adding attendance: {e}")




def get_attendance_data(date_str):
    try:
        file_path = f'Attendance/Attendance-{date_str}.csv'
        if not os.path.exists(file_path):
            return [], [], [], 0
        
        df = pd.read_csv(file_path)
        return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
    except Exception as e:
        print(f"Error fetching attendance data: {e}")
        return [], [], [], 0

def get_all_users():
    try:
        # Ensure we only get directories in static/faces
        return [name for name in os.listdir('static/faces') 
                if os.path.isdir(os.path.join('static/faces', name))]
    except Exception as e:
        print(f"Error getting users: {e}")
        return []


import os
import shutil
import traceback

def deletefolder(folder_path):
    """
    Deletes the specified folder and all its contents.

    Args:
        folder_path (str): The path of the folder to delete.

    Returns:
        bool: True if the folder was successfully deleted, False otherwise.
    """
    try:
        if os.path.exists(folder_path):
            print(f"DEBUG: Attempting to delete folder: {folder_path}")
            
            # Use shutil.rmtree for recursive deletion
            shutil.rmtree(folder_path)
            
            # Verify deletion
            if not os.path.exists(folder_path):
                print(f"DEBUG: Successfully deleted folder: {folder_path}")
                return True
            else:
                print(f"DEBUG: Folder still exists after deletion attempt: {folder_path}")
        else:
            print(f"DEBUG: Folder does not exist: {folder_path}")
            return False
    except Exception as e:
        print(f"Error deleting folder: {traceback.format_exc()}")
        return False



# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/listusers')
def list_users():
    try:
        # Improved user listing with more robust error handling
        users = get_all_users()
        names, rolls = [], []
        
        # Print debug information
        print(f"DEBUG: Total users found: {len(users)}")
        print(f"DEBUG: Users list: {users}")
        
        for user in users:
            # More comprehensive splitting and validation
            try:
                # Split from the right to handle names with multiple underscores
                parts = user.rsplit('_', 1)
                if len(parts) == 2:
                    name, roll = parts
                    names.append(name)
                    rolls.append(roll)
                    print(f"DEBUG: Processed user: Name={name}, Roll={roll}")
                else:
                    print(f"DEBUG: Skipping invalid user format: {user}")
            except Exception as user_error:
                print(f"DEBUG: Error processing user {user}: {user_error}")
        
        # Create attendance file if not exists
        create_attendance_file()
        
        # Get total registered users
        total = total_registered_users()
        
        # Debug print
        print(f"DEBUG: Names: {names}")
        print(f"DEBUG: Rolls: {rolls}")
        print(f"DEBUG: Total: {total}")
        
        # Use zip to pass both lists
        return render_template('register.html', 
                               names=names, 
                               rolls=rolls, 
                               total=total,
                               zip=zip)
    except Exception as e:
        # Comprehensive error logging
        print(f"DEBUG: Error in list_users: {traceback.format_exc()}")
        return render_template('register.html', 
                               names=[], 
                               rolls=[], 
                               total=0, 
                               error=str(e))
@app.route('/adduser', methods=['GET', 'POST'])
def add_user():
    try:
        if request.method == 'POST':
            name = request.form['newusername']
            roll = request.form['newuserid']
            folder_path = os.path.join('static/faces', f'{name}_{roll}')
            os.makedirs(folder_path, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            count = 0
            while count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(folder_path, f'{count}.jpg'), face)
                    count += 1
                cv2.imshow('Capture Faces', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            
            # Train model after adding user
            train_model()
            return render_template('adduser.html', message="User Added Successfully!")
        return render_template('adduser.html')
    except Exception as e:
        print(f"Error in add_user: {traceback.format_exc()}")
        return render_template('adduser.html', message=f"Error: {str(e)}")

@app.route('/attendance', methods=['GET', 'POST'])
def view_attendance():
    try:
        if request.method == 'POST':
            # Convert date format from YYYY-MM-DD to MM_DD_YY
            input_date = request.form['date']
            date_obj = datetime.strptime(input_date, '%Y-%m-%d')
            formatted_date = date_obj.strftime("%m_%d_%y")
            
            names, rolls, times, count = get_attendance_data(formatted_date)
            
            if count == 0:
                return render_template('attendance.html', message="No Records Found for the Selected Date.")
            
            return render_template('attendance.html', 
                                   names=names, 
                                   rolls=rolls, 
                                   times=times, 
                                   count=count, 
                                   zip=zip)
        return render_template('attendance.html')
    except Exception as e:
        print(f"Error in view_attendance: {traceback.format_exc()}")
        return render_template('attendance.html', message=f"Error: {str(e)}")

@app.route('/takeattendance')
def take_attendance():
    try:
        # Ensure model exists before taking attendance
        if not os.path.exists('static/face_recognition_model.pkl'):
            train_model()
        
        cap = cv2.VideoCapture(0)
        model = joblib.load('static/face_recognition_model.pkl')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (50, 50)).flatten()
                try:
                    user_id = model.predict([resized_face])[0]
                    add_attendance(user_id)
                    cv2.putText(frame, f'{user_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                except Exception as e:
                    print(f"Prediction error: {e}")
            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return render_template('attendance.html', message="Attendance Taken Successfully!")
    except Exception as e:
        print(f"Error in take_attendance: {traceback.format_exc()}")
        return render_template('attendance.html', message=f"Error: {str(e)}")

@app.route('/deleteuser/<path:username>', methods=['POST'])
def deleteuser(username):
    try:
        user_folder = os.path.join('static/faces', username)
        print(f"DEBUG: Attempting to delete user folder: {user_folder}")
        
        if os.path.exists(user_folder):
            deletefolder(user_folder)
            print(f"DEBUG: Successfully deleted user folder: {user_folder}")
        
        # Retrain model if other users remain
        if os.listdir('static/faces'):
            train_model()
        else:
            # Remove model if no users remain
            if os.path.exists('static/face_recognition_model.pkl'):
                os.remove('static/face_recognition_model.pkl')
        
        return redirect(url_for('list_users'))
    except Exception as e:
        print(f"DEBUG: Error deleting user: {traceback.format_exc()}")
        return redirect(url_for('list_users'))


if __name__ == '__main__':
    # Ensure Haar Cascade file is available
    haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haarcascade_path):
        print(f"Haar Cascade file not found at {haarcascade_path}")
    
    # Create initial attendance file
    create_attendance_file()
    
    # Run the app
    app.run(debug=True)
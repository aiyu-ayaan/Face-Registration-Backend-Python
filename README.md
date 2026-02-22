# Face Recognition Attendance System

A FastAPI-based backend system for managing attendance using facial recognition with OpenCV.

## Features

- **User Registration**: Register users with their facial data (single front-facing image)
- **Face Authentication**: Authenticate users and log attendance automatically
- **Attendance Logging**: Track attendance records with timestamps and confidence scores
- **SQLite Database**: Lightweight database storage for user data and attendance logs
- **Auto-Model Download**: Face detection and recognition models are downloaded automatically on first run

## Requirements

- Python 3.8+

## Installation

### 1. Create Virtual Environment (Optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

**Note**: On first run, the system will automatically download the required face detection and recognition models (~10MB total).

## API Endpoints

### Health Check
```
GET /
```


### Register User (Form Upload)
```
POST /register
Content-Type: multipart/form-data

Parameters:
- name: string (required) - Full name of the user
- employee_id: string (required) - Unique employee ID
- image: file (required) - Front-facing photo of the user
```

**How to use in Postman:**
1. Set method to `POST` and URL to `http://localhost:8000/register`
2. Go to the **Body** tab, select **form-data**
3. Add fields:
   - `name` (type: Text)
   - `employee_id` (type: Text)
   - `image` (type: File) and choose your image file
4. Send the request

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/register" \
  -F "name=John Doe" \
  -F "employee_id=EMP001" \
  -F "image=@/path/to/photo.jpg"
```


### Authenticate User (Form Upload)
```
POST /authenticate
Content-Type: multipart/form-data

Parameters:
- image: file (required) - Photo to authenticate
```

**How to use in Postman:**
1. Set method to `POST` and URL to `http://localhost:8000/authenticate`
2. Go to the **Body** tab, select **form-data**
3. Add field:
   - `image` (type: File) and choose your image file
4. Send the request

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/authenticate" \
  -F "image=@/path/to/photo.jpg"
```


### Register User (Binary Body with Headers)
```
POST /register-binary
Content-Type: application/octet-stream
Headers:
- name: string (required) - Full name of the user
- employee_id: string (required) - Unique employee ID

Body (binary):
- Raw image bytes (not base64, just the file itself)
```

**How to use in Postman:**
1. Set method to `POST` and URL to `http://localhost:8000/register-binary`
2. Go to the **Headers** tab and add:
  - `name`: (e.g. John Doe)
  - `employee_id`: (e.g. EMP001)
  - `Content-Type`: `application/octet-stream`
3. Go to the **Body** tab, select **binary**
4. Click **Select File** and choose your image file
5. Send the request

### Authenticate User (Binary Body)
```
POST /authenticate-binary
Content-Type: application/octet-stream

Body (binary):
- Raw image bytes (not base64, just the file itself)
```

**How to use in Postman:**
1. Set method to `POST` and URL to `http://localhost:8000/authenticate-binary`
2. Go to the **Body** tab, select **binary**
3. Click **Select File** and choose your image file
4. Send the request


### List Users
```
GET /users
```


### Get Attendance Logs
```
GET /attendance
GET /attendance?employee_id=EMP001
GET /attendance?limit=50
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Database

The system uses SQLite with the following tables:

### Users Table
- `id`: Primary key
- `name`: User's full name
- `employee_id`: Unique employee identifier
- `face_encoding`: Stored facial encoding (binary)
- `created_at`: Registration timestamp
- `updated_at`: Last update timestamp

### Attendance Logs Table
- `id`: Primary key
- `user_id`: Reference to user
- `employee_id`: Employee ID
- `name`: User's name
- `timestamp`: Authentication timestamp
- `confidence`: Match confidence percentage

## Tips for Best Results

1. **Registration**: Use a clear, front-facing photo with good lighting
2. **Authentication**: Ensure the face is clearly visible and well-lit
3. **Image Quality**: Higher resolution images generally provide better accuracy
4. **Single Face**: The system handles multiple faces but works best with single-face images

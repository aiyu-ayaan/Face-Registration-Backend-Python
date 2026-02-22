from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from database import init_db, get_db, User, AttendanceLog
from face_service import FaceRecognitionService

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Attendance System",
    description="Backend API for attendance management using facial recognition.\n\n**Endpoints:**\n- Register user (form-data or binary)\n- Authenticate user (form-data or binary)\n- List users\n- Attendance logs\n\n**Binary endpoints:**\n- Send image as raw binary in body (Postman: Body > binary)\n- For registration, send 'name' and 'employee_id' as headers\n\n**Contact:**\n- Developer: Anshu\n- Email: example@email.com\n\n**License:** MIT",
    version="1.0.0",
    contact={
        "name": "Ayaan",
        "email": "ayaan35200@gmail.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/aiyu-ayaan/Face-Registration-Backend-Python?tab=MIT-1-ov-file"
    }
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Face Recognition Attendance System"}


@app.post("/register")
async def register_user(
    name: str = Form(..., description="Full name of the user"),
    employee_id: str = Form(..., description="Unique employee ID"),
    image: UploadFile = File(..., description="Front-facing photo of the user"),
    db: Session = Depends(get_db)
):
    """
    Register a new user with their facial data for attendance.
    
    - **name**: Full name of the user
    - **employee_id**: Unique employee identifier
    - **image**: A clear front-facing photo of the user
    
    Returns the registered user details on success.
    """
    # Check if employee_id already exists
    existing_user = db.query(User).filter(User.employee_id == employee_id).first()
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail=f"Employee ID '{employee_id}' is already registered"
        )
    
    # Read image bytes
    image_bytes = await image.read()
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )
    
    # Extract face encoding
    face_service = FaceRecognitionService()
    face_encoding = face_service.extract_face_encoding(image_bytes)
    
    if face_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear front-facing photo."
        )
    
    # Convert encoding to bytes for storage
    encoding_bytes = face_service.encoding_to_bytes(face_encoding)
    
    # Create new user
    new_user = User(
        name=name,
        employee_id=employee_id,
        face_encoding=encoding_bytes
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "User registered successfully",
            "data": {
                "id": new_user.id,
                "name": new_user.name,
                "employee_id": new_user.employee_id,
                "created_at": new_user.created_at.isoformat()
            }
        }
    )


@app.post("/authenticate")
async def authenticate_user(
    image: UploadFile = File(..., description="Photo to authenticate"),
    db: Session = Depends(get_db)
):
    """
    Authenticate a user by their face and log attendance.
    
    - **image**: A photo of the person to authenticate
    
    Returns the identified user details and logs their attendance.
    """
    # Read image bytes
    image_bytes = await image.read()
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )
    
    # Extract face encoding from uploaded image
    face_service = FaceRecognitionService()
    unknown_encoding = face_service.extract_face_encoding(image_bytes)
    
    if unknown_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear photo."
        )
    
    # Get all registered users
    users = db.query(User).all()
    
    if not users:
        raise HTTPException(
            status_code=404,
            detail="No registered users found in the system"
        )
    
    # Prepare known encodings list
    known_encodings = []
    user_map = {}
    
    for user in users:
        encoding = face_service.bytes_to_encoding(user.face_encoding)
        known_encodings.append((user.id, encoding))
        user_map[user.id] = user
    
    # Find best match
    match_result = face_service.find_best_match(unknown_encoding, known_encodings)
    
    if match_result is None:
        raise HTTPException(
            status_code=401,
            detail="Face not recognized. User not found in the system."
        )
    
    user_id, confidence = match_result
    matched_user = user_map[user_id]
    
    # Log attendance
    attendance_log = AttendanceLog(
        user_id=matched_user.id,
        employee_id=matched_user.employee_id,
        name=matched_user.name,
        confidence=f"{confidence:.2f}%"
    )
    
    db.add(attendance_log)
    db.commit()
    db.refresh(attendance_log)
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "User authenticated successfully",
            "data": {
                "user": {
                    "id": matched_user.id,
                    "name": matched_user.name,
                    "employee_id": matched_user.employee_id
                },
                "authentication": {
                    "confidence": f"{confidence:.2f}%",
                    "timestamp": attendance_log.timestamp.isoformat(),
                    "attendance_log_id": attendance_log.id
                }
            }
        }
    )



from fastapi import Header, Request

@app.post("/register-binary")
async def register_user_binary(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Register a new user with facial data using binary image data.
    
    - **name**: Full name of the user (send as header)
    - **employee_id**: Unique employee identifier (send as header)
    - **image**: Raw binary image data (body)
    
    Returns the registered user details on success.
    """
    # Accept both lower-case and typical header-case for Postman
    headers = request.headers
    name = headers.get('name') or headers.get('Name')
    employee_id = headers.get('employee_id') or headers.get('Employee-Id') or headers.get('Employee_Id')
    if not name or not employee_id:
        raise HTTPException(
            status_code=400,
            detail="Both 'name' and 'employee_id' headers are required."
        )
    # Check if employee_id already exists
    existing_user = db.query(User).filter(User.employee_id == employee_id).first()
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail=f"Employee ID '{employee_id}' is already registered"
        )
    # Read raw binary body
    image = await request.body()
    # Extract face encoding
    face_service = FaceRecognitionService()
    face_encoding = face_service.extract_face_encoding(image)
    if face_encoding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear front-facing photo."
        )
    # Convert encoding to bytes for storage
    encoding_bytes = face_service.encoding_to_bytes(face_encoding)
    # Create new user
    new_user = User(
        name=name,
        employee_id=employee_id,
        face_encoding=encoding_bytes
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "User registered successfully",
            "data": {
                "id": new_user.id,
                "name": new_user.name,
                "employee_id": new_user.employee_id,
                "created_at": new_user.created_at.isoformat()
            }
        }
    )


@app.post("/authenticate-binary")
async def authenticate_user_binary(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Authenticate a user by their face using binary image data and log attendance.
    
    Body parameters:
    - **image**: Raw binary image data
    
    Returns the identified user details and logs their attendance.
    """
    # Only use the binary body, ignore headers
    # Read raw binary body
    image = await request.body()
    face_service = FaceRecognitionService()
    unknown_encoding = face_service.extract_face_encoding(image)
    if unknown_encoding is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "No face detected in the image. Please upload a clear photo."}
        )
    users = db.query(User).all()
    if not users:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No registered users found in the system."}
        )
    known_encodings = []
    user_map = {}
    for user in users:
        encoding = face_service.bytes_to_encoding(user.face_encoding)
        known_encodings.append((user.id, encoding))
        user_map[user.id] = user
    match_result = face_service.find_best_match(unknown_encoding, known_encodings)
    if match_result is None:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Face not recognized. User not found in the system."}
        )
    user_id, _ = match_result
    matched_user = user_map[user_id]
    return JSONResponse(
        status_code=200,
        content={"success": True, "name": matched_user.name}
    )


@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    """List all registered users (without face encoding data)"""
    users = db.query(User).all()
    
    return {
        "success": True,
        "count": len(users),
        "data": [
            {
                "id": user.id,
                "name": user.name,
                "employee_id": user.employee_id,
                "created_at": user.created_at.isoformat()
            }
            for user in users
        ]
    }


@app.get("/attendance")
def get_attendance_logs(
    employee_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get attendance logs, optionally filtered by employee_id"""
    query = db.query(AttendanceLog)
    
    if employee_id:
        query = query.filter(AttendanceLog.employee_id == employee_id)
    
    logs = query.order_by(AttendanceLog.timestamp.desc()).limit(limit).all()
    
    return {
        "success": True,
        "count": len(logs),
        "data": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "name": log.name,
                "employee_id": log.employee_id,
                "confidence": log.confidence,
                "timestamp": log.timestamp.isoformat()
            }
            for log in logs
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

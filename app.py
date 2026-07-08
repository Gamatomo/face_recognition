import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import insightface
from insightface.app import FaceAnalysis

app = FastAPI(title="Face Recognition AI Microservice")

# Initialize InsightFace model
try:
    print("Loading InsightFace model (buffalo_l)... This may take a while on first run.")
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 means CPU (change to >0 for GPU)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load InsightFace: {e}")

class RegisterRequest(BaseModel):
    images: List[str] # List of base64 encoded images (data:image/jpeg;base64,...)

class VerifyRequest(BaseModel):
    image: str
    target_embedding: List[float]

def decode_base64_image(base64_str: str) -> np.ndarray:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/register_face")
def register_face(req: RegisterRequest):
    if len(req.images) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 images required for registration.")
    
    embeddings = []
    
    for i, b64_img in enumerate(req.images):
        img = decode_base64_image(b64_img)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Image {i+1} is corrupted.")
            
        faces = face_app.get(img)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail=f"No face detected in image {i+1}.")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail=f"Multiple faces detected in image {i+1}. Please ensure only one face is visible.")
            
        embeddings.append(faces[0].embedding)
        
    # Average the 5 embeddings to create a robust profile
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Normalize the averaged embedding
    norm_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    return {
        "success": True,
        "embedding": norm_embedding.tolist()
    }

@app.post("/verify_face")
def verify_face(req: VerifyRequest):
    img = decode_base64_image(req.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Image is corrupted.")
        
    faces = face_app.get(img)
    
    if len(faces) == 0:
        return {"success": False, "match": False, "message": "No face detected."}
        
    # We take the largest face if there are multiple
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
    current_embedding = target_face.embedding
    
    # Calculate Cosine Similarity
    target_vector = np.array(req.target_embedding)
    
    # Normalize vectors just in case
    current_norm = current_embedding / np.linalg.norm(current_embedding)
    target_norm = target_vector / np.linalg.norm(target_vector)
    
    similarity = np.dot(current_norm, target_norm)
    
    # Standard threshold for InsightFace ArcFace lowered to 0.35
    is_match = bool(similarity > 0.35)
    
    return {
        "success": True,
        "match": is_match,
        "similarity": float(similarity)
    }

class IdentifyRequest(BaseModel):
    image: str
    targets: dict[str, List[float]]

@app.post("/identify_face")
def identify_face(req: IdentifyRequest):
    img = decode_base64_image(req.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Image is corrupted.")
        
    faces = face_app.get(img)
    
    if len(faces) == 0:
        return {"success": False, "match": None, "message": "No face detected."}
        
    # We take the largest face if there are multiple
    target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
    current_embedding = target_face.embedding
    current_norm = current_embedding / np.linalg.norm(current_embedding)
    
    best_match = None
    best_similarity = 0.0
    
    for face_key, target_emb in req.targets.items():
        target_vector = np.array(target_emb)
        target_norm = target_vector / np.linalg.norm(target_vector)
        
        similarity = np.dot(current_norm, target_norm)
        
        if similarity > best_similarity:
            best_similarity = float(similarity)
            best_match = face_key
            
    # Threshold for ArcFace lowered for easier detection
    if best_similarity > 0.35:
        return {
            "success": True,
            "match": best_match,
            "similarity": best_similarity
        }
    else:
        return {
            "success": False,
            "match": None,
            "message": "No matching face found."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

from flask import Flask, render_template, request, jsonify, redirect, url_for
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import uuid

app = Flask(__name__)

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded!")

# Initialize Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_APIKEY_HERE")

# Create or connect to index
index_name = "photos"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Create uploads folder
os.makedirs('static/uploads', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['photo']
    if file.filename == '':
        return redirect(url_for('home'))
    
    # Save the uploaded file
    filename = str(uuid.uuid4()) + '.jpg'
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    
    # Convert image to vector using CLIP
    image = Image.open(filepath)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    vector = image_features[0].tolist()
    
    # Store in Pinecone
    index.upsert([{
        "id": filename,
        "values": vector,
        "metadata": {"filename": filename, "filepath": filepath}
    }])
    
    # Redirect back to home page
    return redirect(url_for('home'))

@app.route('/search', methods=['POST'])
def search_photos():
    query = request.form.get('query', '')
    
    # Convert text query to vector
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    query_vector = text_features[0].tolist()
    
    # Search in Pinecone
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    
    # Get only the best match (highest score)
    photos = []
    if results['matches']:
        best_match = results['matches'][0]  # First result is highest score
        photos.append({
            'filename': best_match['metadata']['filename'],
            'score': best_match['score']
        })
    
    return render_template('results.html', photos=photos, query=query)

@app.route('/all_photos')
def all_photos():
    # Get all photos from the uploads folder
    photos = []
    if os.path.exists('static/uploads'):
        for filename in os.listdir('static/uploads'):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                photos.append(filename)
    
    return render_template('all_photos.html', photos=photos)

@app.route('/delete/<filename>')
def delete_photo(filename):
    try:
        # Delete from Pinecone
        index.delete(ids=[filename])
        
        # Delete the actual file
        filepath = os.path.join('static/uploads', filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return redirect(url_for('all_photos'))
    except Exception as e:
        return f"Error deleting photo: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
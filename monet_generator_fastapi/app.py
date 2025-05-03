from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Monet Style Transfer API",
    description="Convert photos to Monet-style paintings using CycleGAN",
    version="1.0.0",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path configuration
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "saved_model"
)  # Use the SavedModel directory instead
SAMPLES_PATH = os.environ.get("SAMPLES_PATH", "samples")

# Mount static files directory if it exists
if os.path.exists(SAMPLES_PATH):
    app.mount("/samples", StaticFiles(directory=SAMPLES_PATH), name="samples")


# Global variable for the model
model = None

# Stats tracking
request_count = 0
processing_times = []


@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model on startup"""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        # Check if directory exists before attempting to load
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model path {MODEL_PATH} does not exist!")
            return

        # For SavedModel directory, we need to use tf.saved_model.load
        if os.path.isdir(MODEL_PATH):
            print("Loading SavedModel directory...")
            model = tf.saved_model.load(MODEL_PATH)
            print("SavedModel loaded successfully!")

            # Detect the signature to use for predictions
            print(f"Available signatures: {list(model.signatures.keys())}")

            # Store signature for prediction
            if "serving_default" in model.signatures:
                model.prediction_function = model.signatures["serving_default"]
                print("Using 'serving_default' signature for predictions")
            else:
                # Use the first available signature
                sig_key = list(model.signatures.keys())[0]
                model.prediction_function = model.signatures[sig_key]
                print(f"Using '{sig_key}' signature for predictions")

            # Test with dummy input - Use the prediction_function instead of calling model directly
            print("Performing warm-up inference...")
            dummy_input = tf.constant(tf.random.normal([1, 256, 256, 3]))
            _ = model.prediction_function(dummy_input)
            print("Model loaded successfully and ready for inference!")
        else:
            # For .h5 or .keras files
            print("Loading Keras model file...")
            model = tf.keras.models.load_model(
                MODEL_PATH, compile=False, custom_objects={"tf": tf}
            )
            print("Keras model loaded successfully!")
            print("Performing warm-up inference...")
            dummy_input = tf.random.normal([1, 256, 256, 3])
            _ = model.predict(dummy_input)
            print("Model loaded successfully and ready for inference!")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback

        traceback.print_exc()
        # Model will remain None, and API will return error when used


def preprocess_image(image):
    """Preprocess image for the model"""
    # Resize to model's expected input
    image = image.resize((256, 256))
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(image)
    img_array = (img_array / 127.5) - 1
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def postprocess_image(img_array):
    """Convert model output back to displayable image"""
    # Denormalize from [-1, 1] to [0, 255]
    img_array = (img_array * 0.5 + 0.5) * 255
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    # Convert to PIL Image
    return Image.fromarray(img_array)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface"""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Monet Style Transfer API</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
        h1 { color: #2c3e50; }
        form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        input, button { margin: 10px 0; }
        .sample { margin: 20px 0; }
        .sample-images { display: flex; margin: 10px 0; }
        .sample-images img { max-width: 300px; margin-right: 10px; border: 1px solid #ddd; }
        code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Monet Style Transfer API</h1>
    <p>This API converts regular photos into Monet-style paintings using a trained CycleGAN model.</p>
    
    <h2>Try it out</h1>
    <form action="/generate/" method="post" enctype="multipart/form-data" target="_blank">
        <p>Upload a photo to convert to Monet style:</p>
        <input type="file" name="file" accept="image/*" required>
        <br>
        <button type="submit">Generate Monet Style</button>
    </form>
    
    <div class="sample">
        <h2>Sample Transformations</h2>
        <div class="sample-images">
            <div>
                <p>Sample 1 (Input)</p>
                <img src="/samples/sample_0_input.jpg" alt="Sample Input 1">
            </div>
            <div>
                <p>Sample 1 (Output)</p>
                <img src="/samples/sample_0_output.jpg" alt="Sample Output 1">
            </div>
        </div>
    </div>
    
    <h2>API Usage</h2>
    <p>Use the following endpoint to generate Monet-style images:</p>
    <code>POST /generate/</code>
    
    <p>Example with curl:</p>
    <code>curl -X POST "http://localhost:8000/generate/" -H "accept: image/jpeg" -H "Content-Type: multipart/form-data" -F "file=@your_photo.jpg" --output monet_style.jpg</code>
    
    <p>The API will return the Monet-style image directly. For programmatic usage, see the <a href="/docs">API documentation</a>.</p>
</body>
</html>
    """
    return html_content


@app.get("/stats")
async def get_stats():
    """Return statistics about API usage"""
    global request_count, processing_times

    avg_time = sum(processing_times) / max(len(processing_times), 1)
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0

    return {
        "total_requests": request_count,
        "average_processing_time_ms": round(avg_time * 1000, 2),
        "max_processing_time_ms": round(max_time * 1000, 2),
        "min_processing_time_ms": round(min_time * 1000, 2),
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
    }


@app.post("/generate/")
async def generate_monet(file: UploadFile = File(...)):
    """Generate a Monet-style image from an uploaded photo"""
    global request_count, processing_times
    request_count += 1

    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please try again later."
        )

    start_time = time.time()

    try:
        # Read and validate image
        content = await file.read()
        image = Image.open(io.BytesIO(content))

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        input_tensor = preprocess_image(image)

        # Convert to TensorFlow tensor
        input_tensor = tf.cast(input_tensor, tf.float32)

        # Different handling based on model type
        if hasattr(model, "prediction_function"):
            # SavedModel with signature
            output_tensor = model.prediction_function(tf.constant(input_tensor))

            # SavedModel signatures return dictionaries, extract the relevant tensor
            output_keys = list(output_tensor.keys())
            print(f"Output keys: {output_keys}")
            output_tensor = output_tensor[output_keys[0]]  # Use first output

        elif hasattr(model, "predict"):
            # Keras model
            output_tensor = model.predict(input_tensor)
        else:
            # Direct callable SavedModel
            output_tensor = model(input_tensor, training=False)

        # Process the output tensor to get the image data
        if isinstance(output_tensor, dict):
            # If still a dict, get the first value
            output_tensor = list(output_tensor.values())[0]

        if isinstance(output_tensor, list):
            output_tensor = output_tensor[0]  # Take first element if it's a list

        # Convert to numpy if still a tensor
        if isinstance(output_tensor, tf.Tensor):
            output_tensor = output_tensor.numpy()

        # Remove batch dimension if present
        if output_tensor.shape[0] == 1:
            output_tensor = output_tensor[0]

        # Postprocess and prepare response
        output_image = postprocess_image(output_tensor)
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)

        # Record processing time
        elapsed = time.time() - start_time
        processing_times.append(elapsed)
        if len(processing_times) > 100:  # Keep only last 100 times
            processing_times = processing_times[-100:]

        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        print(f"Error processing image: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


# Add a JSON endpoint for easier frontend integration
@app.post("/api/generate/")
async def generate_monet_base64(file: UploadFile = File(...)):
    """Generate a Monet-style image and return as base64 for frontend integration"""
    global request_count, processing_times
    request_count += 1

    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please try again later."
        )

    start_time = time.time()

    try:
        # Read and validate image
        content = await file.read()
        image = Image.open(io.BytesIO(content))

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Rest of processing is same as /generate/ endpoint
        input_tensor = preprocess_image(image)
        input_tensor = tf.cast(input_tensor, tf.float32)

        if hasattr(model, "prediction_function"):
            output_tensor = model.prediction_function(tf.constant(input_tensor))
            output_keys = list(output_tensor.keys())
            output_tensor = output_tensor[output_keys[0]]
        elif hasattr(model, "predict"):
            output_tensor = model.predict(input_tensor)
        else:
            output_tensor = model(input_tensor, training=False)

        if isinstance(output_tensor, dict):
            output_tensor = list(output_tensor.values())[0]
        if isinstance(output_tensor, list):
            output_tensor = output_tensor[0]
        if isinstance(output_tensor, tf.Tensor):
            output_tensor = output_tensor.numpy()
        if output_tensor.shape[0] == 1:
            output_tensor = output_tensor[0]

        output_image = postprocess_image(output_tensor)

        # Convert to base64 for JSON response
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)
        import base64

        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Record processing time
        elapsed = time.time() - start_time
        processing_times.append(elapsed)
        if len(processing_times) > 100:
            processing_times = processing_times[-100:]

        # Return JSON response with base64 image and metadata
        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{base64_image}",
            "processing_time_ms": round(elapsed * 1000, 2),
        }

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        print(f"Error processing image: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/sample/{sample_id}")
async def get_sample(sample_id: int):
    """Return a sample transformation pair"""
    input_path = os.path.join(SAMPLES_PATH, f"sample_{sample_id}_input.jpg")
    output_path = os.path.join(SAMPLES_PATH, f"sample_{sample_id}_output.jpg")

    if not os.path.exists(input_path) or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found")

    # Return HTML that displays both images
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample {sample_id}</title>
        <style>
            body {{ display: flex; flex-direction: column; align-items: center; }}
            .image-container {{ display: flex; margin: 20px; }}
            .image-box {{ margin: 10px; text-align: center; }}
            img {{ max-width: 400px; max-height: 400px; }}
        </style>
    </head>
    <body>
        <h1>Sample {sample_id}</h1>
        <div class="image-container">
            <div class="image-box">
                <h2>Original Photo</h2>
                <img src="/samples/sample_{sample_id}_input.jpg" alt="Input">
            </div>
            <div class="image-box">
                <h2>Monet Style</h2>
                <img src="/samples/sample_{sample_id}_output.jpg" alt="Output">
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="localhost", port=port, reload=False)

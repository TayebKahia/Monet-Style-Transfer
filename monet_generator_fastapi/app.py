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
    allow_origins=[
        "http://localhost:3000",  # Next.js local frontend URL
        "https://monet-st-front.vercel.app",  # Your Vercel deployed domain
        "*",  # Allow all origins temporarily for testing
    ],
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
    # Save original size for later use
    original_size = image.size
    # Resize to model's expected input
    image = image.resize((256, 256))
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(image)
    img_array = (img_array / 127.5) - 1
    # Add batch dimension
    return np.expand_dims(img_array, axis=0), original_size


def postprocess_image(img_array, original_size=None):
    """Convert model output back to displayable image"""
    # Denormalize from [-1, 1] to [0, 255]
    img_array = (img_array * 0.5 + 0.5) * 255
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    # Convert to PIL Image
    output_image = Image.fromarray(img_array)
    # Resize back to original dimensions if provided
    if original_size:
        output_image = output_image.resize(original_size, Image.LANCZOS)
    return output_image


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

        # Preprocess image with original size preservation
        input_tensor, original_size = preprocess_image(image)
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

        # Resize back to original dimensions
        output_image = postprocess_image(output_tensor, original_size)

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="localhost", port=port, reload=False)

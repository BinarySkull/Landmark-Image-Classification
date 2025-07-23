from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as T
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

# --- Model and Prediction Logic ---

# 1. Load trained model
# The model is loaded onto the CPU for inference.
model = torch.jit.load("transfer_exported.pt", map_location="cpu")
# Set the model to evaluation mode
model.eval()

# 2. Define the image transformations
# These transforms must be the same as the ones used during model training and export.
# The mean and std values are from 'helpers.py.
mean = [0.4638, 0.4725, 0.4687]
std = [0.2697, 0.2706, 0.3017]

transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# 3. Create a list of class names
# The order of these names must match the order used to train the model.
class_names = [
    '00.Haleakala_National_Park', '01.Mount_Rainier_National_Park', '02.Ljubljana_Castle',
    '03.Dead_Sea', '04.Wroclaws_Dwarves', '05.London_Olympic_Stadium', '06.Niagara_Falls',
    '07.Stonehenge', '08.Grand_Canyon', '09.Golden_Gate_Bridge', '10.Edinburgh_Castle',
    '11.Mount_Rushmore_National_Memorial', '12.Kantanagar_Temple', '13.Yellowstone_National_Park', 
    '14.Terminal_Tower', '15.Central_Park', '16.Eiffel_Tower', '17.Changdeokgung', '18.Delicate_Arch', 
    '19.Vienna_City_Hall', '20.Matterhorn', '21.Taj_Mahal', '22.Moscow_Raceway', '23.Externsteine', 
    '24.Soreq_Cave', '25.Banff_National_Park', '26.Pont_du_Gard', '27.Seattle_Japanese_Garden', 
    '28.Sydney_Harbour_Bridge', '29.Petronas_Towers', '30.Brooklyn_Bridge', '31.Washington_Monument', 
    '32.Hanging_Temple', '33.Sydney_Opera_House', '34.Great_Barrier_Reef', '35.Monumento_a_la_Revolucion', 
    '36.Badlands_National_Park', '37.Atomium', '38.Forth_Bridge', '39.Gateway_of_India', '40.Stockholm_City_Hall', 
    '41.Machu_Picchu', '42.Death_Valley_National_Park', '43.Gullfoss_Falls', '44.Trevi_Fountain', '45.Temple_of_Heaven', 
    '46.Great_Wall_of_China', '47.Prague_Astronomical_Clock', '48.Whitby_Abbey', '49.Temple_of_Olympian_Zeus'
]


def predict(image_bytes: bytes):
    """
    Takes image bytes, applies transforms, and returns the predicted landmark.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))

    # Apply the transformations and add a batch dimension
    image_tensor = transforms(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        # The output is a tensor of probabilities. We get the index of the highest one.
        _, predicted_idx = torch.max(outputs, 1)

    # Return the predicted class name
    return class_names[predicted_idx.item()]


# --- FastAPI Endpoints ---

@app.get("/")
def read_root():
    """
    Root endpoint with a welcome message.
    """
    return {"message": "Welcome to the Landmark Prediction API"}


@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Prediction endpoint. Receives an image file and returns the predicted landmark.
    """
    # Read the contents of the uploaded file
    image_bytes = await file.read()

    # Get the prediction
    prediction = predict(image_bytes)

    # Return the result
    return {"filename": file.filename, "prediction": prediction}
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import pandas as pd
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

from process import preprocess_files  # your preprocessing function
from custom_model import HybridCNN_MLP_Bayesian  # your model class

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 43  # Adjust based on your input features

model = HybridCNN_MLP_Bayesian(input_dim=input_dim).to(device)
model.load_state_dict(torch.load('model\hybrid_model.pkl', map_location=device))
model.eval()

# CORS settings (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ThreadPoolExecutor for running blocking preprocessing
executor = ThreadPoolExecutor(max_workers=1)

async def run_preprocessing(*args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, preprocess_files, *args)

async def save_to_tempfile(upload_file: UploadFile) -> str:
    temp = tempfile.NamedTemporaryFile(delete=False)
    content = await upload_file.read()
    temp.write(content)
    temp.flush()
    temp.close()
    return temp.name

@app.post("/predict")
async def predict(
    logon_file: UploadFile = File(...),
    email_file: UploadFile = File(...),
    file_file: UploadFile = File(...),
    device_file: UploadFile = File(...),
    http_file: UploadFile = File(...),
    psych_file: UploadFile = File(...)
):
    try:
        # Save files to temp locations
        logon_path = await save_to_tempfile(logon_file)
        email_path = await save_to_tempfile(email_file)
        file_path = await save_to_tempfile(file_file)
        device_path = await save_to_tempfile(device_file)
        http_path = await save_to_tempfile(http_file)
        psych_path = await save_to_tempfile(psych_file)

        # Load CSVs
        logon = pd.read_csv(logon_path, nrows=500000)
        email = pd.read_csv(email_path, nrows=500000)
        file = pd.read_csv(file_path, nrows=500000)
        device = pd.read_csv(device_path, nrows=500000)
        http = pd.read_csv(http_path, nrows=3000000)
        psych = pd.read_csv(psych_path)

        # Preprocess
        processed_df = await run_preprocessing(logon, email, file, device, http, psych)

        # Prepare tensor inputs (dropping user or non-feature columns)
        features = processed_df.drop(columns=['user']).values
        inputs = torch.tensor(features, dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs).cpu().numpy().tolist()

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running!"}

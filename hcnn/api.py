import io
import yaml
import math
import torch
import librosa
import torchaudio
import numpy as np
from typing import List, Tuple, Dict
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from hcnn.model import HarmonicFeatureExtractor, RagaClassifier, RagaClassifierConfig, HarmonicFeatureExtractorConfig, AudioTransformerConfig

class HCNNPredictor:
    def __init__(self, config_path: str, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model configurations
        self.fe_config = HarmonicFeatureExtractorConfig(
            sample_rate=self.config['model']['fe_config']['sample_rate'],
            n_fft=self.config['model']['fe_config']['n_fft'],
            hop_length=self.config['model']['fe_config']['hop_length'],
            n_harmonics=self.config['model']['fe_config']['n_harmonics'],
            n_filters_per_semitone=self.config['model']['fe_config']['n_filters_per_semitone'],
        )
        
        self.be_config = AudioTransformerConfig(
            d_model=self.config['model']['be_config']['d_model'],
            nhead=self.config['model']['be_config']['nhead'],
            num_layers=self.config['model']['be_config']['num_layers'],
            dim_feedforward=self.config['model']['be_config']['dim_feedforward'],
            dropout=self.config['model']['be_config']['dropout'],
            num_filters=self.config['model']['be_config']['num_filters'],
            num_classes=self.config['model']['be_config']['num_classes'],
        )
        
        # Create model
        self.model_config = RagaClassifierConfig(
            fe_config=self.fe_config,
            be_config=self.be_config
        )
        self.model = RagaClassifier(config=self.model_config)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Store labels
        self.labels = self.config['dataset']['labels']
        
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> torch.Tensor:
        """Preprocess audio data for model input."""
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)
        
        # Resample if necessary
        if sr != self.fe_config.sample_rate:
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sr, 
                target_sr=self.fe_config.sample_rate
            )
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Split into frames
        audio_tensor = audio_tensor[None, ...] # channel first
        segment_len = int(self.fe_config.target_length * self.fe_config.sample_rate)
        num_segments = math.ceil(audio_tensor.shape[-1] / segment_len)
        if audio_tensor.shape[-1] < num_segments * segment_len:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (num_segments * segment_len-audio_tensor.shape[-1]))
        audio_tensor = audio_tensor.view(num_segments, 1, segment_len)

        # Define timestamps
        timestamps = torch.linspace(0, num_segments, steps=1, dtype=torch.float32, device=self.device)*self.fe_config.target_length
        timestamps += self.fe_config.target_length/2
        
        return audio_tensor, timestamps
    
    def predict(self, audio_data: np.ndarray, sr: int) -> Tuple[Dict[str, float], float]:
        """Make prediction on audio data."""
        with torch.no_grad():
            # Preprocess audio
            audio_tensor, timestamps = self.preprocess_audio(audio_data, sr).to(self.device)

            # Get model prediction
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1).detach().numpy()
            timestamps = timestamps.detach().numpy()
            
            # Create prob dictionary for each timestamp
            transcription = {
                'timetamps' : timestamps , 
                'probabilities' : probabilities , 
                'predicted_class' : [self.labels[x] for x in np.argmax(probabilities, axis=-1)],
            }
            
            return transcription

# Initialize FastAPI app
app = FastAPI(title="HCNN Raga Classification API")

# Initialize predictor
predictor = None

@app.lifespan("startup")
async def startup_event():
    global predictor
    predictor = HCNNPredictor(
        config_path="logs/250517-170038-hcnn-experiment-4RagaClf-exp2/config.yaml",
        model_path="logs/250517-170038-hcnn-experiment-4RagaClf-exp2/best_model.pth"
    )

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Read audio file
        contents = await file.read()
        audio_data, sr = librosa.load(io.BytesIO(contents), sr=None)
        
        # Get prediction
        transcription = predictor.predict(audio_data, sr)
        
        return JSONResponse({
            "info": predictor.config,
            "transcription": transcription,
            "status": "success"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "status": "error"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
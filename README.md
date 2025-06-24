# Stream Recommender System

A machine learning-powered educational stream recommendation system that predicts the most suitable academic stream (Science, Commerce, Arts, or Vocational) for students based on their academic performance and career aspirations.

## 🎯 Overview

This project uses a deep neural network (ANN) to analyze student academic scores across multiple subjects and their career aspirations to recommend the most appropriate academic stream. The system achieves 94% validation accuracy and includes an interactive Streamlit web application for easy use.

## 🚀 Features

- **High Accuracy**: 94% validation accuracy on test data
- **Multi-Subject Analysis**: Considers performance in Math, Physics, Biology, Chemistry, English, History, and Geography
- **Career Integration**: Incorporates career aspirations into recommendations
- **Interactive Web App**: Easy-to-use Streamlit interface
- **Model Explainability**: LIME integration for prediction explanations
- **Robust Architecture**: 4-layer neural network with dropout regularization

## 📊 Model Performance

- **Validation Accuracy**: 94%
- **Training Epochs**: 200
- **Architecture**: 4-layer ANN (128→64→32→16→4 neurons)
- **Regularization**: 30% dropout on each layer
- **Loss Function**: Cross-entropy with class weighting

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stream-recommender.git
   cd stream-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Place your `student-scores-with-stream.csv` file in the `data/` directory

## 📁 Project Structure

```
stream-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── student-scores-with-stream.csv
├── train.py              # Model training script   
├── app/
│   ├── streamlit_app.py      # Streamlit web application
│   ├── model.pth             # Trained model weights
│   ├── scaler.pkl            # Feature scaler
│   ├── career_ohe.pkl        # Career aspiration encoder
│   ├── label_encoder.pkl     # Stream label encoder
│   └── background.npy        # LIME background data
├── notebooks/
│   └── Stream_predictor.ipynb # Exploratory analysis and development
```

## 🏃‍♂️ Usage

### Training the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train the neural network
- Save the trained model and preprocessors to `app/` directory
- Display training progress and final accuracy

### Running the Web Application

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Using the Model Programmatically

```python
import torch
from src.model import StreamANN, load_model
from src.preprocess import preprocess_input, load_preprocessors

# Load model and preprocessors
model = load_model('app/model.pth', input_dim=24)
scaler, career_ohe, label_encoder = load_preprocessors(
    'app/scaler.pkl', 
    'app/career_ohe.pkl', 
    'app/label_encoder.pkl'
)

# Prepare input data
numeric_features = [85, 90, 78, 82, 88, 75, 80]  # Subject scores
career_aspiration = "Engineering"

# Preprocess and predict
processed_input = preprocess_input(numeric_features, career_aspiration, scaler, career_ohe)
input_tensor = torch.tensor(processed_input, dtype=torch.float32)

with torch.no_grad():
    prediction = model(input_tensor)
    predicted_stream = label_encoder.inverse_transform([torch.argmax(prediction, dim=1).item()])[0]

print(f"Recommended Stream: {predicted_stream}")
```

## 📊 Dataset

The model is trained on student academic data including:

- **Subject Scores** (0-100): Math, Physics, Biology, Chemistry, English, History, Geography
- **Career Aspirations**: Various career fields (Engineering, Medicine, Business, etc.)
- **Target Streams**: Science, Commerce, Arts, Vocational

### Data Preprocessing

- Numeric features are standardized using StandardScaler
- Career aspirations are one-hot encoded
- Class imbalance is handled using balanced class weights

## 🔍 Model Explainability

The application includes LIME (Local Interpretable Model-agnostic Explanations) to provide insights into why specific recommendations are made. This helps users understand which factors most influenced the prediction.

## 🧪 Testing

Run the notebook `notebooks/Stream_predictor.ipynb` to see the complete development process, including:
- Data exploration
- Model development
- Performance evaluation
- SHAP value analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:
- `torch`: Neural network framework
- `scikit-learn`: Preprocessing and metrics
- `pandas`: Data manipulation
- `streamlit`: Web application framework
- `shap`: Model explainability
- `lime`: Local explanations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Thanks to the contributors of the student academic dataset
- PyTorch community for excellent documentation
- Streamlit team for the amazing web framework

## 📈 Future Improvements

- [ ] Add more diverse datasets for better generalization
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add support for additional demographic factors
- [ ] Create REST API for programmatic access
- [ ] Add automated model retraining pipeline
- [ ] Implement A/B testing framework for model versions

## 🐛 Known Issues

- Large model files require Git LFS for version control
- SHAP explanations may take time to load in the web interface

## 📞 Support

If you have any questions or run into issues, please [open an issue](https://github.com/yourusername/stream-recommender/issues) on GitHub.

---

**Made with ❤️ for better educational guidance**
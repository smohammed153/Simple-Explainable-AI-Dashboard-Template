# Clinical AI Dashboard for Clinical Decision Support

An interactive Streamlit-based dashboard that demonstrates explainable AI for clinical decision making. This template provides a user-friendly interface for healthcare professionals to interact with AI models and understand their predictions.

## Features

- Interactive Dashboard: Real-time prediction updates
- Model Explainability: Feature importance visualization
- Clinical Parameters: Adjustable sliders for patient data
- Risk Assessment: Clear visualization of risk factors
- Responsive Design: Works on desktop and tablet devices

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn (for enhanced visualizations)

## Installation Guide

### 1. Prerequisites
- Python 3.7 or later
- pip (Python package manager)

### 2. Setting Up a Virtual Environment (Recommended)

# Create a virtual environment
python -m venv dashboard_env

# Activate the environment
# Windows:
.\dashboard_env\Scripts\activate
# macOS/Linux:
# source dashboard_env/bin/activate

### 3. Install Required Packages

# Core requirements
pip install streamlit pandas numpy matplotlib scikit-learn seaborn

# For development with Jupyter Notebook (optional)
pip install notebook

# For code formatting (optional)
pip install black flake8

### 4. Verify Installation

streamlit --version
python -c "import pandas as pd; import streamlit as st; print('All packages installed successfully!')"

## Running the Dashboard

1. Navigate to the project directory:
      cd path/to/project
   

2. Run the Streamlit app:
      streamlit run "Simple Explainable AI Dashboard Template.py"
   

3. Access the dashboard:
   - The app will automatically open in your default web browser
   - If not, navigate to: http://localhost:8501

## Dashboard Features

### Sidebar
- Adjust clinical parameters using sliders
- Click "Analyze Patient" to process the input

### Main Panel
- Clinical Assessment: Shows prediction results and confidence levels
- Decision Factors: Displays feature importance and patient values
- Visualization: Interactive charts showing how different features affect the prediction

## Clinical Parameters

The dashboard works with the following clinical features:

1. Vital Signs
   - Age (years)
   - Systolic Blood Pressure (mmHg)
   - Diastolic Blood Pressure (mmHg)
   - Heart Rate (bpm)
   - Body Temperature (°C)

2. Lab Results
   - Oxygen Saturation (%)
   - Glucose Level (mg/dL)
   - White Blood Cell Count (thousands/μL)
   - Hemoglobin (g/dL)
   - BMI (kg/m²)

## Customization

### Changing the Model
To use a different scikit-learn model:

# In the train_model method
from sklearn.ensemble import GradientBoostingClassifier
self.model = GradientBoostingClassifier()
self.model.fit(X_train, y_train)

### Adding New Features
1. Add new sliders in the sidebar
2. Update the feature processing logic
3. Extend the visualization components

## Troubleshooting

- Port in use: If port 8501 is busy, specify a different port:
    streamlit run app.py --server.port 8502
  

- Missing packages: Ensure all required packages are installed:
    pip install -r requirements.txt
  

- Browser issues: Try accessing via http://localhost:8501 if the browser doesn't open automatically

## Deployment Options

### Local Deployment
For team access on a local network:
streamlit run app.py --server.port 8501 --server.address=0.0.0.0

### Cloud Deployment
1. Streamlit Cloud: Deploy directly from GitHub
2. Docker: Containerize the application
3. AWS/GCP: Deploy on cloud platforms

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## Support

For support, please open an issue in the GitHub repository or contact the development team.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample medical data
@st.cache_data
def create_sample_data():
    X, y = make_classification(n_samples=1000, n_features=10, 
                              n_informative=5, n_redundant=2,
                              random_state=42)
    feature_names = [
        'Age',
        'Systolic_BP',
        'Diastolic_BP',
        'Heart_Rate',
        'Body_Temp',
        'O2_Saturation',
        'Glucose_Level',
        'WBC_Count',
        'Hemoglobin',
        'BMI'
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df['Diagnosis'] = y
    return df, feature_names

def main():
    st.title("Clinical AI Co-pilot Prototype")
    st.write("Demonstration of explainable AI for clinical decision support")
    
    # Load data
    df, features = create_sample_data()
    
    # Sidebar for user input
    st.sidebar.header("Patient Parameters")
    selected_features = {}
    for feature in features[:5]:  # Show first 5 features for demo
        selected_features[feature] = st.sidebar.slider(
            feature, float(df[feature].min()), float(df[feature].max()),
            float(df[feature].mean())
        )
    
    # Model training (in real scenario, this would be pre-trained)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df['Diagnosis'])
    
    # Prediction and explanation
    if st.sidebar.button('Analyze Patient'):
        input_data = np.array([list(selected_features.values()) + [0]*5]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Clinical Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Diagnosis", 
                     "High Risk" if prediction == 1 else "Low Risk",
                     delta=f"{probability[1]*100:.1f}% confidence")
        
        with col2:
            st.metric("Recommendation", 
                     "Further Investigation Recommended" if prediction == 1 
                     else "Routine Monitoring")
        
        # Feature importance for this prediction
        st.subheader("Decision Factors")
        importance_df = pd.DataFrame({
            'Feature': list(selected_features.keys()),
            'Importance': model.feature_importances_[:5],
            'Patient_Value': list(selected_features.values())
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Factors Influencing This Decision')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
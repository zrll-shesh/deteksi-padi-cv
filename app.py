# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# import json
# import os
# from datetime import datetime

# # ==================== PAGE CONFIGURATION ====================
# st.set_page_config(
#     page_title="AgroSight - Rice Disease Detection",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==================== CUSTOM CSS ====================
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
#     * {
#         font-family: 'Poppins', sans-serif;
#     }
    
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#     }
    
#     .stApp {
#         background: transparent;
#     }
    
#     .hero-section {
#         background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
#         padding: 3rem;
#         border-radius: 20px;
#         box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#         text-align: center;
#         margin-bottom: 2rem;
#         animation: fadeInDown 1s ease;
#     }
    
#     .hero-title {
#         font-size: 3.5rem;
#         font-weight: 700;
#         color: white;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
#         margin-bottom: 0.5rem;
#         letter-spacing: 2px;
#     }
    
#     .hero-subtitle {
#         font-size: 1.3rem;
#         color: #e8f5e9;
#         font-weight: 300;
#         margin-top: 0;
#     }
    
#     .card {
#         background: white;
#         padding: 2rem;
#         border-radius: 15px;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#         margin-bottom: 1.5rem;
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#         animation: fadeInUp 0.8s ease;
#     }
    
#     .card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 15px 40px rgba(0,0,0,0.3);
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 12px;
#         text-align: center;
#         box-shadow: 0 8px 20px rgba(0,0,0,0.2);
#         transition: transform 0.3s ease;
#     }
    
#     .metric-card:hover {
#         transform: scale(1.05);
#     }
    
#     .metric-value {
#         font-size: 2.5rem;
#         font-weight: 700;
#         margin: 0.5rem 0;
#     }
    
#     .metric-label {
#         font-size: 1rem;
#         font-weight: 300;
#         opacity: 0.9;
#     }
    
#     .prediction-card {
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#         margin: 1rem 0;
#     }
    
#     .prediction-title {
#         font-size: 1.8rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#     }
    
#     .prediction-disease {
#         font-size: 2.5rem;
#         font-weight: 700;
#         text-transform: uppercase;
#         letter-spacing: 2px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
#     }
    
#     .confidence-bar {
#         background: rgba(255,255,255,0.3);
#         border-radius: 10px;
#         padding: 0.5rem;
#         margin-top: 1rem;
#     }
    
#     .disease-info {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 12px;
#         margin-top: 1rem;
#     }
    
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.8rem 2rem;
#         font-size: 1.1rem;
#         font-weight: 600;
#         border-radius: 25px;
#         cursor: pointer;
#         transition: all 0.3s ease;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
#         width: 100%;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.3);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
#         color: white;
#     }
    
#     .upload-section {
#         border: 3px dashed #667eea;
#         border-radius: 15px;
#         padding: 2rem;
#         text-align: center;
#         background: rgba(102, 126, 234, 0.1);
#         transition: all 0.3s ease;
#     }
    
#     .upload-section:hover {
#         border-color: #764ba2;
#         background: rgba(118, 75, 162, 0.1);
#     }
    
#     @keyframes fadeInDown {
#         from {
#             opacity: 0;
#             transform: translateY(-30px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     @keyframes fadeInUp {
#         from {
#             opacity: 0;
#             transform: translateY(30px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2rem;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background: white;
#         border-radius: 10px;
#         padding: 1rem 2rem;
#         font-weight: 600;
#         box-shadow: 0 4px 10px rgba(0,0,0,0.1);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#     }
    
#     .footer {
#         text-align: center;
#         padding: 2rem;
#         background: rgba(255,255,255,0.1);
#         border-radius: 15px;
#         margin-top: 3rem;
#         color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ==================== MODEL CONFIGURATION ====================
# CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut']
# MODEL_PATH = 'models/best_model.pth'
# IMG_SIZE = 224

# DISEASE_INFO = {
#     'Bacterialblight': {
#         'name': 'Bacterial Blight',
#         'scientific': 'Xanthomonas oryzae pv. oryzae',
#         'symptoms': [
#             'Water-soaked lesions on leaf margins',
#             'Yellow to white lesions with wavy edges',
#             'Systemic infection leads to wilting',
#             'Bacterial ooze visible in morning'
#         ],
#         'treatment': [
#             'Use resistant varieties',
#             'Apply copper-based bactericides',
#             'Maintain field sanitation',
#             'Avoid excessive nitrogen fertilizer'
#         ],
#         'severity': 'HIGH',
#         'color': '#ef5350'
#     },
#     'Brownspot': {
#         'name': 'Brown Spot',
#         'scientific': 'Bipolaris oryzae',
#         'symptoms': [
#             'Circular brown spots with gray centers',
#             'Spots surrounded by yellow halos',
#             'Numerous spots cause leaf blight',
#             'Affects yield and grain quality'
#         ],
#         'treatment': [
#             'Apply fungicides (Mancozeb, Tricyclazole)',
#             'Balance nutrient management',
#             'Ensure proper drainage',
#             'Use disease-free seeds'
#         ],
#         'severity': 'MEDIUM',
#         'color': '#ff9800'
#     },
#     'Leafsmut': {
#         'name': 'Leaf Smut',
#         'scientific': 'Entyloma oryzae',
#         'symptoms': [
#             'Small black angular spots on leaves',
#             'Spots aligned between leaf veins',
#             'Severe infection causes leaf yellowing',
#             'Reduces photosynthesis efficiency'
#         ],
#         'treatment': [
#             'Apply systemic fungicides early',
#             'Remove infected plant debris',
#             'Crop rotation with non-host crops',
#             'Improve air circulation in field'
#         ],
#         'severity': 'LOW-MEDIUM',
#         'color': '#66bb6a'
#     }
# }

# # ==================== MODEL LOADING ====================
# @st.cache_resource
# def load_model():
#     """Load the trained ResNet-34 model"""
#     class ResNet34RiceClassifier(nn.Module):
#         def __init__(self, num_classes=3):
#             super(ResNet34RiceClassifier, self).__init__()
#             self.model = models.resnet34(pretrained=False)
#             num_features = self.model.fc.in_features
#             self.model.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_features, 512),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(512),
#                 nn.Dropout(0.3),
#                 nn.Linear(512, num_classes)
#             )
        
#         def forward(self, x):
#             return self.model(x)
    
#     model = ResNet34RiceClassifier(num_classes=len(CLASSES))
#     if os.path.exists(MODEL_PATH):
#         model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#         model.eval()
#         return model
#     return None

# @st.cache_resource
# def load_metrics():
#     """Load training metrics"""
#     metrics_path = 'results/metrics.json'
#     if os.path.exists(metrics_path):
#         with open(metrics_path, 'r') as f:
#             return json.load(f)
#     return None

# # ==================== IMAGE PREPROCESSING ====================
# def preprocess_image(image):
#     """Preprocess image for model input"""
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return transform(image).unsqueeze(0)

# # ==================== PREDICTION ====================
# def predict_disease(model, image):
#     """Predict disease from image"""
#     with torch.no_grad():
#         image_tensor = preprocess_image(image)
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output, dim=1)[0]
#         predicted_class = torch.argmax(probabilities).item()
#         confidence = probabilities[predicted_class].item()
        
#         results = {
#             'class': CLASSES[predicted_class],
#             'confidence': confidence,
#             'all_probabilities': {
#                 CLASSES[i]: float(probabilities[i]) 
#                 for i in range(len(CLASSES))
#             }
#         }
#         return results

# # ==================== VISUALIZATION FUNCTIONS ====================
# def create_confidence_gauge(confidence):
#     """Create confidence gauge chart"""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number+delta",
#         value=confidence * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Confidence Score", 'font': {'size': 24, 'color': 'white'}},
#         delta={'reference': 80, 'increasing': {'color': "green"}},
#         gauge={
#             'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
#             'bar': {'color': "darkblue"},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 50], 'color': '#ffcdd2'},
#                 {'range': [50, 75], 'color': '#fff9c4'},
#                 {'range': [75, 100], 'color': '#c8e6c9'}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 90
#             }
#         }
#     ))
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': "white", 'family': "Poppins"},
#         height=300
#     )
#     return fig

# def create_probability_chart(probabilities):
#     """Create probability bar chart"""
#     df = pd.DataFrame(list(probabilities.items()), 
#                      columns=['Disease', 'Probability'])
#     df['Probability'] = df['Probability'] * 100
    
#     fig = px.bar(
#         df, x='Probability', y='Disease',
#         orientation='h',
#         color='Probability',
#         color_continuous_scale='Viridis',
#         labels={'Probability': 'Probability (%)'},
#         title='Prediction Probabilities for All Classes'
#     )
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(255,255,255,0.9)',
#         font={'family': 'Poppins', 'size': 12},
#         title_font={'size': 18, 'color': '#333'},
#         height=400,
#         showlegend=False
#     )
    
#     fig.update_traces(
#         marker_line_color='rgb(8,48,107)',
#         marker_line_width=2,
#         texttemplate='%{x:.2f}%',
#         textposition='outside'
#     )
    
#     return fig

# # ==================== MAIN APP ====================
# def main():
#     # Hero Section
#     st.markdown("""
#     <div class="hero-section">
#         <h1 class="hero-title">AgroSight</h1>
#         <p class="hero-subtitle">AI-Powered Rice Disease Detection System</p>
#         <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">
#             Empowering farmers with cutting-edge deep learning technology
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load model
#     model = load_model()
#     metrics = load_metrics()
    
#     if model is None:
#         st.error("Model not found! Please train the model first using train_model.py")
#         return
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("### Navigation")
#         page = st.radio("", 
#             ["Disease Detection", "Model Performance", "Disease Information", "About"],
#             label_visibility="collapsed"
#         )
        
#         st.markdown("---")
#         st.markdown("### Quick Stats")
#         if metrics:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Accuracy", f"{metrics['test_accuracy']*100:.1f}%")
#             with col2:
#                 st.metric("Classes", len(CLASSES))
        
#         st.markdown("---")
#         st.markdown("""
#         <div style='text-align: center; padding: 1rem;'>
#             <p style='color: white; font-size: 0.9rem;'>
#                 Developed with ResNet-34
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Main Content
#     if page == "Disease Detection":
#         show_detection_page(model)
#     elif page == "Model Performance":
#         show_performance_page(metrics)
#     elif page == "Disease Information":
#         show_disease_info_page()
#     else:
#         show_about_page()

# def show_detection_page(model):
#     """Disease detection page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Upload Rice Leaf Image")
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown('<div class="upload-section">', unsafe_allow_html=True)
#         uploaded_file = st.file_uploader(
#             "Choose an image...",
#             type=['jpg', 'jpeg', 'png'],
#             help="Upload a clear image of rice leaf"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file).convert('RGB')
#             st.image(image, caption='Uploaded Image', use_container_width=True)
            
#             if st.button("Analyze Image", key="analyze"):
#                 with st.spinner("Analyzing..."):
#                     results = predict_disease(model, image)
                    
#                     st.session_state['results'] = results
#                     st.session_state['analyzed'] = True
    
#     with col2:
#         if 'analyzed' in st.session_state and st.session_state['analyzed']:
#             results = st.session_state['results']
#             disease = results['class']
#             confidence = results['confidence']
            
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <div class="prediction-title">Prediction Result</div>
#                 <div class="prediction-disease">{disease}</div>
#                 <div style="font-size: 1.2rem; margin-top: 0.5rem;">
#                     Confidence: {confidence*100:.2f}%
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.plotly_chart(
#                 create_confidence_gauge(confidence),
#                 use_container_width=True
#             )
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Show detailed results
#     if 'analyzed' in st.session_state and st.session_state['analyzed']:
#         results = st.session_state['results']
        
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("## Detailed Analysis")
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.plotly_chart(
#                 create_probability_chart(results['all_probabilities']),
#                 use_container_width=True
#             )
        
#         with col2:
#             disease_data = DISEASE_INFO[results['class']]
            
#             st.markdown(f"""
#             <div class="disease-info">
#                 <h3>{disease_data['name']}</h3>
#                 <p><strong>Scientific Name:</strong> {disease_data['scientific']}</p>
#                 <p><strong>Severity:</strong> {disease_data['severity']}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.markdown("### Key Symptoms")
#             for symptom in disease_data['symptoms']:
#                 st.markdown(f"- {symptom}")
            
#             st.markdown("### Recommended Treatment")
#             for treatment in disease_data['treatment']:
#                 st.markdown(f"- {treatment}")
        
#         st.markdown('</div>', unsafe_allow_html=True)

# def show_performance_page(metrics):
#     """Model performance page"""
#     if metrics is None:
#         st.warning("Metrics not available. Please train the model first.")
#         return
    
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Model Performance Metrics")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Overall Accuracy</div>
#             <div class="metric-value">{metrics['test_accuracy']*100:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         avg_precision = np.mean([
#             metrics['classification_report'][cls]['precision'] 
#             for cls in CLASSES
#         ])
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Avg Precision</div>
#             <div class="metric-value">{avg_precision*100:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         avg_recall = np.mean([
#             metrics['classification_report'][cls]['recall'] 
#             for cls in CLASSES
#         ])
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Avg Recall</div>
#             <div class="metric-value">{avg_recall*100:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Classification Report
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Per-Class Performance")
    
#     class_metrics = []
#     for cls in CLASSES:
#         class_metrics.append({
#             'Disease': cls,
#             'Precision': f"{metrics['classification_report'][cls]['precision']*100:.2f}%",
#             'Recall': f"{metrics['classification_report'][cls]['recall']*100:.2f}%",
#             'F1-Score': f"{metrics['classification_report'][cls]['f1-score']*100:.2f}%",
#             'Support': metrics['classification_report'][cls]['support']
#         })
    
#     df = pd.DataFrame(class_metrics)
#     st.dataframe(df, use_container_width=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Visualization Images
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Training Visualizations")
    
#     viz_files = {
#         'Training History': 'results/training_history.png',
#         'Confusion Matrix': 'results/confusion_matrix.png',
#         'ROC Curves': 'results/roc_curves.png',
#         'Class Performance': 'results/class_performance.png'
#     }
    
#     for title, path in viz_files.items():
#         if os.path.exists(path):
#             st.markdown(f"### {title}")
#             st.image(path, use_container_width=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_disease_info_page():
#     """Disease information page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Rice Disease Information")
    
#     for disease, info in DISEASE_INFO.items():
#         with st.expander(f"**{info['name']}** ({info['scientific']})", expanded=True):
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 st.markdown("### Symptoms")
#                 for symptom in info['symptoms']:
#                     st.markdown(f"- {symptom}")
                
#                 st.markdown(f"**Severity Level:** {info['severity']}")
            
#             with col2:
#                 st.markdown("### Treatment Recommendations")
#                 for treatment in info['treatment']:
#                     st.markdown(f"- {treatment}")
            
#             st.markdown("---")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_about_page():
#     """About page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## About AgroSight")
    
#     st.markdown("""
#     **AgroSight** is an advanced AI-powered platform designed to detect rice leaf diseases 
#     using deep learning technology. The system employs a ResNet-34 convolutional neural 
#     network architecture to classify three major rice diseases with high accuracy.
    
#     ### Key Features
#     - **Real-time Detection**: Upload images and get instant disease predictions
#     - **High Accuracy**: Trained on thousands of rice leaf images
#     - **Detailed Information**: Comprehensive disease symptoms and treatments
#     - **User-Friendly**: Simple interface accessible to farmers and agronomists
    
#     ### Technology Stack
#     - **Deep Learning**: ResNet-34 architecture
#     - **Framework**: PyTorch
#     - **Interface**: Streamlit
#     - **Visualization**: Plotly, Matplotlib
    
#     ### Diseases Detected
#     1. **Bacterial Blight** - Caused by *Xanthomonas oryzae*
#     2. **Brown Spot** - Caused by *Bipolaris oryzae*
#     3. **Leaf Smut** - Caused by *Entyloma oryzae*
    
#     ### Mission
#     Our mission is to support sustainable agriculture and strengthen national food security 
#     by providing farmers with accessible, accurate, and timely disease detection tools.
    
#     ### Contact
#     For more information, feedback, or collaboration opportunities, please reach out through 
#     the feedback form or contact your local agricultural extension service.
#     """)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <p style="font-size: 1.1rem; font-weight: 600;">
#             AgroSight - Empowering Precision Agriculture
#         </p>
#         <p style="font-size: 0.9rem; margin-top: 0.5rem;">
#             Developed with PyTorch & Streamlit | 2025
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# import json
# import os
# from datetime import datetime
# import cv2
# import io

# # ==================== PAGE CONFIGURATION ====================
# st.set_page_config(
#     page_title="AgroSight - Rice Disease Detection",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==================== CUSTOM CSS ====================
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
#     * {
#         font-family: 'Poppins', sans-serif;
#         margin: 0;
#         padding: 0;
#         box-sizing: border-box;
#     }
    
#     html, body {
#         margin: 0;
#         padding: 0;
#         overflow-x: hidden;
#     }
    
#     .main {
#         background: linear-gradient(135deg, #0c2461 0%, #1e3799 20%, #4a69bd 100%);
#         min-height: 100vh;
#         padding: 0;
#     }
    
#     .stApp {
#         background: transparent;
#         padding-top: 0;
#     }
    
#     /* Hero Section */
#     .hero-section {
#         background: linear-gradient(135deg, 
#             rgba(26, 188, 156, 0.95) 0%, 
#             rgba(22, 160, 133, 0.95) 30%,
#             rgba(19, 141, 117, 0.95) 100%);
#         padding: 4rem 2rem;
#         border-radius: 0 0 30px 30px;
#         box-shadow: 0 20px 60px rgba(0,0,0,0.4);
#         text-align: center;
#         margin-bottom: 3rem;
#         animation: fadeInDown 1s ease-out;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .hero-section::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         bottom: 0;
#         background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100"><path fill="rgba(255,255,255,0.1)" d="M0,50 Q250,100 500,50 T1000,50 V100 H0 Z"/></svg>');
#         background-size: cover;
#         opacity: 0.3;
#     }
    
#     .hero-title {
#         font-size: 4rem;
#         font-weight: 800;
#         color: white;
#         text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
#         margin-bottom: 0.5rem;
#         letter-spacing: 3px;
#         background: linear-gradient(45deg, #ffffff 30%, #f0f0f0 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     .hero-subtitle {
#         font-size: 1.5rem;
#         color: #e8f8f5;
#         font-weight: 400;
#         margin-top: 0;
#         text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
#         max-width: 800px;
#         margin: 0 auto;
#         line-height: 1.6;
#     }
    
#     /* Cards */
#     .card {
#         background: rgba(255, 255, 255, 0.98);
#         backdrop-filter: blur(20px);
#         -webkit-backdrop-filter: blur(20px);
#         padding: 2.5rem;
#         border-radius: 25px;
#         box-shadow: 
#             0 15px 35px rgba(0,0,0,0.2),
#             0 5px 15px rgba(0,0,0,0.1),
#             inset 0 1px 0 rgba(255,255,255,0.5);
#         margin-bottom: 2rem;
#         transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
#         border: 1px solid rgba(255,255,255,0.3);
#         animation: fadeInUp 0.8s ease-out;
#     }
    
#     .card:hover {
#         transform: translateY(-10px) scale(1.01);
#         box-shadow: 
#             0 25px 50px rgba(0,0,0,0.3),
#             0 15px 25px rgba(0,0,0,0.2),
#             inset 0 1px 0 rgba(255,255,255,0.6);
#     }
    
#     /* Metric Cards */
#     .metric-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#         gap: 1.5rem;
#         margin: 2rem 0;
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
#         color: white;
#         padding: 2rem 1.5rem;
#         border-radius: 20px;
#         text-align: center;
#         box-shadow: 
#             0 10px 30px rgba(0,0,0,0.3),
#             0 4px 15px rgba(0,0,0,0.2);
#         transition: all 0.3s ease;
#         position: relative;
#         overflow: hidden;
#         border: 2px solid rgba(255,255,255,0.2);
#     }
    
#     .metric-card::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
#         transition: 0.5s;
#     }
    
#     .metric-card:hover::before {
#         left: 100%;
#     }
    
#     .metric-card:hover {
#         transform: translateY(-5px) scale(1.05);
#         box-shadow: 
#             0 20px 40px rgba(0,0,0,0.4),
#             0 10px 25px rgba(0,0,0,0.3);
#     }
    
#     .metric-value {
#         font-size: 3rem;
#         font-weight: 800;
#         margin: 0.5rem 0;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#         background: linear-gradient(45deg, #ffffff 30%, #e0e0e0 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     .metric-label {
#         font-size: 1.1rem;
#         font-weight: 500;
#         opacity: 0.9;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     /* Prediction Card */
#     .prediction-card {
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#         color: white;
#         padding: 2.5rem;
#         border-radius: 25px;
#         box-shadow: 
#             0 20px 40px rgba(0,0,0,0.3),
#             0 10px 25px rgba(0,0,0,0.2);
#         margin: 1.5rem 0;
#         animation: pulse 2s infinite;
#         border: 3px solid rgba(255,255,255,0.3);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .prediction-card::after {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
#         background-size: 20px 20px;
#         opacity: 0.3;
#         animation: moveBackground 20s linear infinite;
#     }
    
#     .prediction-title {
#         font-size: 2rem;
#         font-weight: 700;
#         margin-bottom: 1.5rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
#         position: relative;
#         z-index: 1;
#     }
    
#     .prediction-disease {
#         font-size: 3.5rem;
#         font-weight: 800;
#         text-transform: uppercase;
#         letter-spacing: 3px;
#         text-shadow: 
#             3px 3px 6px rgba(0,0,0,0.3),
#             0 0 20px rgba(255,255,255,0.5);
#         margin: 1rem 0;
#         position: relative;
#         z-index: 1;
#         background: linear-gradient(45deg, #ffffff 30%, #f0f0f0 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     /* Disease Info */
#     .disease-info {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 20px;
#         margin-top: 1.5rem;
#         box-shadow: 0 15px 30px rgba(0,0,0,0.2);
#         border: 2px solid rgba(255,255,255,0.3);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .disease-info::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         height: 5px;
#         background: linear-gradient(90deg, 
#             #ff0000, #ff9900, #ffff00, #00ff00, 
#             #00ffff, #0000ff, #9900ff);
#         background-size: 200% 100%;
#         animation: rainbow 3s linear infinite;
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 1rem 2.5rem;
#         font-size: 1.2rem;
#         font-weight: 700;
#         border-radius: 50px;
#         cursor: pointer;
#         transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
#         box-shadow: 
#             0 10px 25px rgba(0,0,0,0.2),
#             0 5px 15px rgba(0,0,0,0.1);
#         position: relative;
#         overflow: hidden;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#     }
    
#     .stButton > button::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
#         transition: 0.5s;
#     }
    
#     .stButton > button:hover::before {
#         left: 100%;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-3px) scale(1.05);
#         box-shadow: 
#             0 20px 40px rgba(0,0,0,0.3),
#             0 15px 25px rgba(0,0,0,0.2);
#         background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
#     }
    
#     /* Sidebar */
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, 
#             rgba(26, 35, 126, 0.95) 0%, 
#             rgba(48, 63, 159, 0.95) 100%) !important;
#         backdrop-filter: blur(20px);
#         -webkit-backdrop-filter: blur(20px);
#         color: white;
#         border-right: 2px solid rgba(255,255,255,0.1);
#     }
    
#     /* Upload Section */
#     .upload-section {
#         border: 4px dashed rgba(102, 126, 234, 0.8);
#         border-radius: 25px;
#         padding: 3rem 2rem;
#         text-align: center;
#         background: rgba(102, 126, 234, 0.15);
#         backdrop-filter: blur(10px);
#         transition: all 0.4s ease;
#         min-height: 300px;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         align-items: center;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .upload-section::before {
#         content: '';
#         position: absolute;
#         width: 200px;
#         height: 200px;
#         background: rgba(255,255,255,0.1);
#         border-radius: 50%;
#         animation: float 6s ease-in-out infinite;
#     }
    
#     .upload-section:hover {
#         border-color: #764ba2;
#         background: rgba(118, 75, 162, 0.2);
#         transform: translateY(-5px);
#         box-shadow: 0 20px 40px rgba(0,0,0,0.15);
#     }
    
#     /* Camera Section */
#     .camera-section {
#         border: 4px dashed rgba(38, 166, 154, 0.8);
#         border-radius: 25px;
#         padding: 2rem;
#         text-align: center;
#         background: rgba(38, 166, 154, 0.15);
#         backdrop-filter: blur(10px);
#         transition: all 0.4s ease;
#         min-height: 300px;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .camera-section::before {
#         content: '';
#         position: absolute;
#         width: 200px;
#         height: 200px;
#         background: rgba(255,255,255,0.1);
#         border-radius: 50%;
#         animation: float 6s ease-in-out infinite reverse;
#     }
    
#     .camera-section:hover {
#         border-color: #26a69a;
#         background: rgba(38, 166, 154, 0.25);
#         transform: translateY(-5px);
#         box-shadow: 0 20px 40px rgba(0,0,0,0.15);
#     }
    
#     /* Camera Container */
#     .camera-container {
#         width: 100%;
#         height: 400px;
#         border-radius: 20px;
#         overflow: hidden;
#         box-shadow: 0 15px 35px rgba(0,0,0,0.3);
#         margin: 1.5rem 0;
#         position: relative;
#         background: #000;
#     }
    
#     .camera-preview {
#         width: 100%;
#         height: 100%;
#         object-fit: cover;
#         transition: transform 0.3s ease;
#     }
    
#     .camera-controls {
#         position: absolute;
#         bottom: 20px;
#         left: 0;
#         right: 0;
#         display: flex;
#         justify-content: center;
#         gap: 1rem;
#         z-index: 10;
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2rem;
#         padding: 1rem 0;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background: rgba(255, 255, 255, 0.9);
#         border-radius: 15px;
#         padding: 1rem 2rem;
#         font-weight: 700;
#         box-shadow: 0 8px 20px rgba(0,0,0,0.15);
#         transition: all 0.3s ease;
#         border: 2px solid transparent;
#         font-size: 1.1rem;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 12px 25px rgba(0,0,0,0.2);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border-color: rgba(255,255,255,0.5);
#         box-shadow: 0 15px 30px rgba(0,0,0,0.25);
#     }
    
#     /* Progress Bar */
#     .confidence-bar {
#         background: rgba(255,255,255,0.2);
#         border-radius: 15px;
#         padding: 0.8rem;
#         margin-top: 1.5rem;
#         position: relative;
#         overflow: hidden;
#         border: 2px solid rgba(255,255,255,0.3);
#     }
    
#     .confidence-fill {
#         position: absolute;
#         top: 0;
#         left: 0;
#         height: 100%;
#         background: linear-gradient(90deg, #4CAF50, #8BC34A);
#         border-radius: 13px;
#         transition: width 1s ease-in-out;
#         box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
#     }
    
#     /* Footer */
#     .footer {
#         text-align: center;
#         padding: 3rem;
#         background: rgba(0, 0, 0, 0.3);
#         backdrop-filter: blur(20px);
#         border-radius: 25px;
#         margin-top: 4rem;
#         color: white;
#         border: 1px solid rgba(255,255,255,0.1);
#         box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#     }
    
#     /* Animations */
#     @keyframes fadeInDown {
#         from {
#             opacity: 0;
#             transform: translateY(-50px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     @keyframes fadeInUp {
#         from {
#             opacity: 0;
#             transform: translateY(50px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     @keyframes pulse {
#         0% {
#             box-shadow: 0 20px 40px rgba(0,0,0,0.3);
#         }
#         50% {
#             box-shadow: 0 25px 50px rgba(0,0,0,0.4);
#         }
#         100% {
#             box-shadow: 0 20px 40px rgba(0,0,0,0.3);
#         }
#     }
    
#     @keyframes moveBackground {
#         from {
#             transform: rotate(0deg);
#         }
#         to {
#             transform: rotate(360deg);
#         }
#     }
    
#     @keyframes float {
#         0%, 100% {
#             transform: translateY(0px) translateX(0px);
#         }
#         25% {
#             transform: translateY(-20px) translateX(20px);
#         }
#         50% {
#             transform: translateY(0px) translateX(40px);
#         }
#         75% {
#             transform: translateY(20px) translateX(20px);
#         }
#     }
    
#     @keyframes rainbow {
#         0% {
#             background-position: 0% 50%;
#         }
#         100% {
#             background-position: 200% 50%;
#         }
#     }
    
#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(135deg, #667eea, #764ba2);
#         border-radius: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(135deg, #764ba2, #667eea);
#     }
    
#     /* Selection */
#     ::selection {
#         background: rgba(102, 126, 234, 0.5);
#         color: white;
#     }
    
#     /* Loading Spinner */
#     .stSpinner > div {
#         border-color: #667eea transparent transparent transparent !important;
#     }
    
#     /* Tooltip */
#     .tooltip {
#         position: relative;
#         display: inline-block;
#     }
    
#     .tooltip .tooltiptext {
#         visibility: hidden;
#         width: 200px;
#         background-color: rgba(0, 0, 0, 0.8);
#         color: #fff;
#         text-align: center;
#         border-radius: 6px;
#         padding: 10px;
#         position: absolute;
#         z-index: 1;
#         bottom: 125%;
#         left: 50%;
#         margin-left: -100px;
#         opacity: 0;
#         transition: opacity 0.3s;
#         font-size: 0.9rem;
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255,255,255,0.1);
#     }
    
#     .tooltip:hover .tooltiptext {
#         visibility: visible;
#         opacity: 1;
#     }
    
#     /* Empty State */
#     .empty-state {
#         text-align: center;
#         padding: 4rem 2rem;
#         color: #666;
#     }
    
#     .empty-state h3 {
#         font-size: 1.8rem;
#         margin-bottom: 1rem;
#         color: #444;
#     }
    
#     .empty-state p {
#         font-size: 1.1rem;
#         opacity: 0.8;
#     }
    
#     /* Disease Severity Badge */
#     .severity-badge {
#         display: inline-block;
#         padding: 0.5rem 1rem;
#         border-radius: 25px;
#         font-weight: 700;
#         text-transform: uppercase;
#         font-size: 0.9rem;
#         letter-spacing: 1px;
#         margin-left: 1rem;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#     }
    
#     .severity-high {
#         background: linear-gradient(135deg, #ff416c, #ff4b2b);
#         color: white;
#     }
    
#     .severity-medium {
#         background: linear-gradient(135deg, #f7971e, #ffd200);
#         color: #333;
#     }
    
#     .severity-low {
#         background: linear-gradient(135deg, #11998e, #38ef7d);
#         color: white;
#     }
    
#     /* History Card */
#     .history-card {
#         background: rgba(255, 255, 255, 0.9);
#         border-radius: 15px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#         border-left: 5px solid #667eea;
#         transition: all 0.3s ease;
#     }
    
#     .history-card:hover {
#         transform: translateX(5px);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.15);
#     }
    
#     /* Responsive Design */
#     @media (max-width: 768px) {
#         .hero-title {
#             font-size: 2.5rem;
#         }
        
#         .hero-subtitle {
#             font-size: 1.2rem;
#         }
        
#         .card {
#             padding: 1.5rem;
#         }
        
#         .metric-card {
#             padding: 1.5rem 1rem;
#         }
        
#         .metric-value {
#             font-size: 2.2rem;
#         }
        
#         .prediction-disease {
#             font-size: 2.5rem;
#         }
        
#         .stTabs [data-baseweb="tab"] {
#             padding: 0.8rem 1.5rem;
#             font-size: 1rem;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# # ==================== MODEL CONFIGURATION ====================
# CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut']
# MODEL_PATH = 'models/best_model.pth'
# IMG_SIZE = 224

# DISEASE_INFO = {
#     'Bacterialblight': {
#         'name': 'Bacterial Blight',
#         'scientific': 'Xanthomonas oryzae pv. oryzae',
#         'symptoms': [
#             'Water-soaked lesions on leaf margins',
#             'Yellow to white lesions with wavy edges',
#             'Systemic infection leads to wilting',
#             'Bacterial ooze visible in morning'
#         ],
#         'treatment': [
#             'Use resistant varieties',
#             'Apply copper-based bactericides',
#             'Maintain field sanitation',
#             'Avoid excessive nitrogen fertilizer'
#         ],
#         'severity': 'HIGH',
#         'color': '#ef5350'
#     },
#     'Brownspot': {
#         'name': 'Brown Spot',
#         'scientific': 'Bipolaris oryzae',
#         'symptoms': [
#             'Circular brown spots with gray centers',
#             'Spots surrounded by yellow halos',
#             'Numerous spots cause leaf blight',
#             'Affects yield and grain quality'
#         ],
#         'treatment': [
#             'Apply fungicides (Mancozeb, Tricyclazole)',
#             'Balance nutrient management',
#             'Ensure proper drainage',
#             'Use disease-free seeds'
#         ],
#         'severity': 'MEDIUM',
#         'color': '#ff9800'
#     },
#     'Leafsmut': {
#         'name': 'Leaf Smut',
#         'scientific': 'Entyloma oryzae',
#         'symptoms': [
#             'Small black angular spots on leaves',
#             'Spots aligned between leaf veins',
#             'Severe infection causes leaf yellowing',
#             'Reduces photosynthesis efficiency'
#         ],
#         'treatment': [
#             'Apply systemic fungicides early',
#             'Remove infected plant debris',
#             'Crop rotation with non-host crops',
#             'Improve air circulation in field'
#         ],
#         'severity': 'LOW-MEDIUM',
#         'color': '#66bb6a'
#     }
# }

# # ==================== MODEL LOADING ====================
# @st.cache_resource
# def load_model():
#     """Load the trained ResNet-34 model"""
#     class ResNet34RiceClassifier(nn.Module):
#         def __init__(self, num_classes=3):
#             super(ResNet34RiceClassifier, self).__init__()
#             self.model = models.resnet34(weights=None)  # Fixed deprecation warning
#             num_features = self.model.fc.in_features
#             self.model.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_features, 512),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(512),
#                 nn.Dropout(0.3),
#                 nn.Linear(512, num_classes)
#             )
        
#         def forward(self, x):
#             return self.model(x)
    
#     model = ResNet34RiceClassifier(num_classes=len(CLASSES))
#     if os.path.exists(MODEL_PATH):
#         model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#         model.eval()
#         return model
#     return None

# @st.cache_resource
# def load_metrics():
#     """Load training metrics"""
#     metrics_path = 'results/metrics.json'
#     if os.path.exists(metrics_path):
#         with open(metrics_path, 'r') as f:
#             return json.load(f)
#     return None

# # ==================== IMAGE PREPROCESSING ====================
# def preprocess_image(image):
#     """Preprocess image for model input"""
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return transform(image).unsqueeze(0)

# # ==================== CAMERA FUNCTIONS ====================
# def capture_from_camera():
#     """Capture image from webcam"""
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         st.error("Cannot access camera. Please check your camera permissions.")
#         return None
    
#     ret, frame = cap.read()
#     cap.release()
    
#     if ret:
#         # Convert BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         return Image.fromarray(frame_rgb)
#     return None

# def process_camera_image(image):
#     """Process camera image for display"""
#     if image is None:
#         return None
    
#     # Convert PIL Image to numpy array
#     if isinstance(image, Image.Image):
#         img_array = np.array(image)
#     else:
#         img_array = image
    
#     if img_array is None or img_array.size == 0:
#         return None
    
#     try:
#         # Add some image enhancement
#         # Apply CLAHE for contrast enhancement
#         lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         l = clahe.apply(l)
#         lab = cv2.merge((l, a, b))
#         enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
#         # Add subtle vignette effect
#         rows, cols = enhanced.shape[:2]
#         kernel_x = cv2.getGaussianKernel(cols, cols/3)
#         kernel_y = cv2.getGaussianKernel(rows, rows/3)
#         kernel = kernel_y * kernel_x.T
#         mask = kernel / kernel.max()
        
#         for i in range(3):
#             enhanced[:,:,i] = enhanced[:,:,i] * mask
        
#         return Image.fromarray(np.clip(enhanced, 0, 255).astype('uint8'))
#     except Exception as e:
#         st.warning(f"Image processing error: {e}")
#         return image  # Return original image if processing fails

# # ==================== PREDICTION ====================
# def predict_disease(model, image):
#     """Predict disease from image"""
#     with torch.no_grad():
#         image_tensor = preprocess_image(image)
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output, dim=1)[0]
#         predicted_class = torch.argmax(probabilities).item()
#         confidence = probabilities[predicted_class].item()
        
#         results = {
#             'class': CLASSES[predicted_class],
#             'confidence': confidence,
#             'all_probabilities': {
#                 CLASSES[i]: float(probabilities[i]) 
#                 for i in range(len(CLASSES))
#             },
#             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         return results

# # ==================== VISUALIZATION FUNCTIONS ====================
# def create_confidence_gauge(confidence):
#     """Create confidence gauge chart"""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number+delta",
#         value=confidence * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Confidence Score", 'font': {'size': 24, 'color': 'white'}},
#         delta={'reference': 80, 'increasing': {'color': "green"}},
#         gauge={
#             'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
#             'bar': {'color': "darkblue"},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 50], 'color': '#ffcdd2'},
#                 {'range': [50, 75], 'color': '#fff9c4'},
#                 {'range': [75, 100], 'color': '#c8e6c9'}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 90
#             }
#         }
#     ))
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': "white", 'family': "Poppins"},
#         height=300
#     )
#     return fig

# def create_probability_chart(probabilities):
#     """Create probability bar chart"""
#     df = pd.DataFrame(list(probabilities.items()), 
#                      columns=['Disease', 'Probability'])
#     df['Probability'] = df['Probability'] * 100
    
#     fig = px.bar(
#         df, x='Probability', y='Disease',
#         orientation='h',
#         color='Probability',
#         color_continuous_scale='Viridis',
#         labels={'Probability': 'Probability (%)'},
#         title='Prediction Probabilities for All Classes'
#     )
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(255,255,255,0.9)',
#         font={'family': 'Poppins', 'size': 12},
#         title_font={'size': 18, 'color': '#333'},
#         height=400,
#         showlegend=False
#     )
    
#     fig.update_traces(
#         marker_line_color='rgb(8,48,107)',
#         marker_line_width=2,
#         texttemplate='%{x:.2f}%',
#         textposition='outside'
#     )
    
#     return fig

# def create_disease_severity_chart():
#     """Create disease severity comparison chart"""
#     disease_names = [DISEASE_INFO[disease]['name'] for disease in CLASSES]
#     severities = []
#     colors = []
#     for disease in CLASSES:
#         info = DISEASE_INFO[disease]
#         severity_value = {
#             'HIGH': 90,
#             'MEDIUM': 60,
#             'LOW-MEDIUM': 40
#         }.get(info['severity'], 50)
#         severities.append(severity_value)
#         colors.append(info['color'])
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=disease_names,
#             y=severities,
#             marker_color=colors,
#             text=[f"{severities[i]}%" for i in range(len(severities))],
#             textposition='auto',
#         )
#     ])
    
#     fig.update_layout(
#         title='Disease Severity Comparison',
#         xaxis_title='Disease',
#         yaxis_title='Severity Level',
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(255,255,255,0.9)',
#         font={'family': 'Poppins'},
#         height=400
#     )
    
#     return fig

# # ==================== MAIN APP ====================
# def main():
#     # Hero Section
#     st.markdown("""
#     <div class="hero-section">
#         <h1 class="hero-title">AgroSight</h1>
#         <p class="hero-subtitle">AI-Powered Rice Disease Detection System</p>
#         <p style="color: white; font-size: 1.1rem; margin-top: 1.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
#             Advanced deep learning technology for accurate and real-time rice disease identification
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load model
#     model = load_model()
#     metrics = load_metrics()
    
#     if model is None:
#         st.error("Model not found! Please train the model first using train_model.py")
#         return
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("### Navigation")
#         page = st.radio("Select Page", 
#             ["Disease Detection", "Model Performance", "Disease Information", "Analysis History", "About"],
#             label_visibility="visible"
#         )
        
#         st.markdown("---")
#         st.markdown("### System Status")
        
#         # Status indicators
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown(f"""
#             <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
#                 <div class="metric-label">Model Status</div>
#                 <div class="metric-value" style="font-size: 1.5rem; color: #4CAF50;">Active</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown(f"""
#             <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
#                 <div class="metric-label">Classes</div>
#                 <div class="metric-value" style="font-size: 1.5rem;">{len(CLASSES)}</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         if metrics:
#             st.markdown("### Performance Metrics")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
#             with col2:
#                 avg_f1 = np.mean([metrics['classification_report'][cls]['f1-score'] for cls in CLASSES]) if 'classification_report' in metrics else 0
#                 st.metric("Avg F1-Score", f"{avg_f1*100:.1f}%")
        
#         st.markdown("---")
#         st.markdown("### Quick Actions")
#         if st.button("Clear Analysis History", key="clear_history"):
#             if 'history' in st.session_state:
#                 st.session_state.history = []
#                 st.success("History cleared successfully!")
        
#         st.markdown("---")
#         st.markdown("""
#         <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;'>
#             <p style='color: white; font-size: 0.9rem; margin-bottom: 0.5rem;'>
#                 System Version: 2.0.1
#             </p>
#             <p style='color: white; font-size: 0.8rem; opacity: 0.8;'>
#                 Powered by ResNet-34
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Main Content
#     if page == "Disease Detection":
#         show_detection_page(model)
#     elif page == "Model Performance":
#         show_performance_page(metrics)
#     elif page == "Disease Information":
#         show_disease_info_page()
#     elif page == "Analysis History":
#         show_history_page()
#     else:
#         show_about_page()

# def show_detection_page(model):
#     """Disease detection page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Upload Rice Leaf Image")
    
#     # Create tabs for different input methods
#     tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Capture from Camera"])
    
#     with tab1:
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.markdown('<div class="upload-section">', unsafe_allow_html=True)
#             uploaded_file = st.file_uploader(
#                 "Choose an image file",
#                 type=['jpg', 'jpeg', 'png'],
#                 help="Upload a clear image of rice leaf for analysis"
#             )
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             if uploaded_file is not None:
#                 image = Image.open(uploaded_file).convert('RGB')
#                 st.image(image, caption='Uploaded Image', use_container_width=True)
                
#                 if st.button("Analyze Uploaded Image", key="analyze_upload"):
#                     with st.spinner("Analyzing image..."):
#                         results = predict_disease(model, image)
                        
#                         # Add to history
#                         if 'history' not in st.session_state:
#                             st.session_state.history = []
#                         st.session_state.history.append({
#                             **results,
#                             'source': 'upload',
#                             'filename': uploaded_file.name if hasattr(uploaded_file, 'name') else 'Uploaded Image',
#                             'image': image
#                         })
                        
#                         st.session_state['results'] = results
#                         st.session_state['analyzed'] = True
#                         st.rerun()
        
#         with col2:
#             if 'analyzed' in st.session_state and st.session_state.get('analyzed'):
#                 results = st.session_state.get('results', {})
#                 if results:
#                     disease = results.get('class', 'Unknown')
#                     confidence = results.get('confidence', 0)
                    
#                     st.markdown(f"""
#                     <div class="prediction-card">
#                         <div class="prediction-title">Detection Result</div>
#                         <div class="prediction-disease">{disease}</div>
#                         <div style="font-size: 1.2rem; margin-top: 1rem;">
#                             Confidence Level: {confidence*100:.2f}%
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.plotly_chart(
#                         create_confidence_gauge(confidence),
#                         use_container_width=True
#                     )
    
#     with tab2:
#         st.markdown('<div class="camera-section">', unsafe_allow_html=True)
#         st.markdown("### Real-time Camera Capture")
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             if st.button("Capture Image", key="capture"):
#                 with st.spinner("Accessing camera..."):
#                     captured_image = capture_from_camera()
#                     if captured_image:
#                         st.session_state['captured_image'] = captured_image
#                         st.session_state['captured_image_display'] = captured_image
#                         st.success("Image captured successfully!")
#                     else:
#                         st.error("Failed to capture image. Please check your camera.")
        
#         with col2:
#             if 'captured_image_display' in st.session_state:
#                 try:
#                     processed_image = process_camera_image(st.session_state['captured_image_display'])
#                     if processed_image:
#                         st.image(processed_image, caption='Captured Image', use_container_width=True)
                        
#                         if st.button("Analyze Captured Image", key="analyze_capture"):
#                             if 'captured_image' in st.session_state:
#                                 with st.spinner("Analyzing captured image..."):
#                                     results = predict_disease(model, st.session_state['captured_image'])
                                    
#                                     # Add to history
#                                     if 'history' not in st.session_state:
#                                         st.session_state.history = []
#                                     st.session_state.history.append({
#                                         **results,
#                                         'source': 'camera',
#                                         'filename': 'Camera Capture',
#                                         'image': st.session_state['captured_image']
#                                     })
                                    
#                                     st.session_state['results'] = results
#                                     st.session_state['analyzed'] = True
#                                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Error displaying image: {e}")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Show detailed results
#     if 'analyzed' in st.session_state and st.session_state.get('analyzed'):
#         results = st.session_state.get('results', {})
#         if results:
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown("## Detailed Analysis Report")
            
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 if 'all_probabilities' in results:
#                     st.plotly_chart(
#                         create_probability_chart(results['all_probabilities']),
#                         use_container_width=True
#                     )
                
#                 # Disease severity visualization
#                 disease_data = DISEASE_INFO.get(results.get('class', ''), {})
#                 if disease_data:
#                     severity = disease_data.get('severity', 'UNKNOWN')
#                     severity_class = {
#                         'HIGH': 'severity-high',
#                         'MEDIUM': 'severity-medium',
#                         'LOW-MEDIUM': 'severity-low'
#                     }.get(severity, 'severity-medium')
                    
#                     st.markdown("### Disease Severity")
#                     st.markdown(f"""
#                     <div style="display: flex; align-items: center; margin: 1rem 0;">
#                         <span style="font-size: 1.2rem; font-weight: 600;">Severity Level:</span>
#                         <span class="severity-badge {severity_class}">
#                             {severity}
#                         </span>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 disease_data = DISEASE_INFO.get(results.get('class', ''), {})
#                 if disease_data:
#                     st.markdown(f"""
#                     <div class="disease-info">
#                         <h3>{disease_data.get('name', 'Unknown Disease')}</h3>
#                         <p><strong>Scientific Name:</strong> {disease_data.get('scientific', 'Unknown')}</p>
#                         <p><strong>Confidence Score:</strong> {results.get('confidence', 0)*100:.2f}%</p>
#                         <p><strong>Detection Time:</strong> {results.get('timestamp', 'Unknown')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("### Key Symptoms")
#                     for symptom in disease_data.get('symptoms', []):
#                         st.markdown(f"""
#                         <div class="history-card" style="padding: 1rem; margin-bottom: 0.5rem;">
#                             <div style="display: flex; align-items: start;">
#                                 <div style="margin-right: 0.5rem; color: #ff4444;">•</div>
#                                 <div>{symptom}</div>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     st.markdown("### Recommended Treatment")
#                     for treatment in disease_data.get('treatment', []):
#                         st.markdown(f"""
#                         <div class="history-card" style="padding: 1rem; margin-bottom: 0.5rem; border-left-color: #4CAF50;">
#                             <div style="display: flex; align-items: start;">
#                                 <div style="margin-right: 0.5rem; color: #4CAF50;">✓</div>
#                                 <div>{treatment}</div>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)

# def show_performance_page(metrics):
#     """Model performance page"""
#     if metrics is None:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("## Model Performance Dashboard")
#         st.info("Metrics not available. Please train the model first to see performance data.")
#         st.markdown('</div>', unsafe_allow_html=True)
#         return
    
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Model Performance Dashboard")
    
#     # Overall Metrics
#     st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         accuracy = metrics.get('test_accuracy', 0) * 100
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Accuracy</div>
#             <div class="metric-value">{accuracy:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         avg_precision = np.mean([
#             metrics['classification_report'][cls]['precision'] 
#             for cls in CLASSES
#         ]) * 100
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Precision</div>
#             <div class="metric-value">{avg_precision:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         avg_recall = np.mean([
#             metrics['classification_report'][cls]['recall'] 
#             for cls in CLASSES
#         ]) * 100
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Recall</div>
#             <div class="metric-value">{avg_recall:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         avg_f1 = np.mean([
#             metrics['classification_report'][cls]['f1-score'] 
#             for cls in CLASSES
#         ]) * 100
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">F1-Score</div>
#             <div class="metric-value">{avg_f1:.2f}%</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Classification Report
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Per-Class Performance Analysis")
    
#     class_metrics = []
#     for cls in CLASSES:
#         class_metrics.append({
#             'Disease': DISEASE_INFO[cls]['name'],
#             'Precision': f"{metrics['classification_report'][cls]['precision']*100:.2f}%",
#             'Recall': f"{metrics['classification_report'][cls]['recall']*100:.2f}%",
#             'F1-Score': f"{metrics['classification_report'][cls]['f1-score']*100:.2f}%",
#             'Support': metrics['classification_report'][cls]['support']
#         })
    
#     df = pd.DataFrame(class_metrics)
#     st.dataframe(df, use_container_width=True, height=250)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Visualizations
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Model Visualizations")
    
#     cols = st.columns(2)
    
#     with cols[0]:
#         # Create accuracy visualization
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=['Bacterial Blight', 'Brown Spot', 'Leaf Smut'],
#             y=[metrics['classification_report'][cls]['precision']*100 for cls in CLASSES],
#             mode='lines+markers',
#             name='Precision',
#             line=dict(color='#667eea', width=3)
#         ))
#         fig.add_trace(go.Scatter(
#             x=['Bacterial Blight', 'Brown Spot', 'Leaf Smut'],
#             y=[metrics['classification_report'][cls]['recall']*100 for cls in CLASSES],
#             mode='lines+markers',
#             name='Recall',
#             line=dict(color='#764ba2', width=3)
#         ))
        
#         fig.update_layout(
#             title='Precision & Recall by Disease',
#             xaxis_title='Disease',
#             yaxis_title='Score (%)',
#             height=400,
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(255,255,255,0.9)'
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with cols[1]:
#         # Create disease severity chart
#         st.plotly_chart(create_disease_severity_chart(), use_container_width=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_disease_info_page():
#     """Disease information page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Rice Disease Information Library")
    
#     for disease, info in DISEASE_INFO.items():
#         with st.expander(f"**{info['name']}** - {info['scientific']}", expanded=False):
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 st.markdown("### Disease Characteristics")
                
#                 severity_class = {
#                     'HIGH': 'severity-high',
#                     'MEDIUM': 'severity-medium',
#                     'LOW-MEDIUM': 'severity-low'
#                 }.get(info['severity'], 'severity-medium')
                
#                 st.markdown(f"""
#                 <div style="margin-bottom: 1.5rem;">
#                     <div style="display: flex; align-items: center; margin-bottom: 1rem;">
#                         <span style="font-weight: 600; margin-right: 0.5rem;">Severity:</span>
#                         <span class="severity-badge {severity_class}">{info['severity']}</span>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("#### Symptoms")
#                 for symptom in info['symptoms']:
#                     st.markdown(f"""
#                     <div class="history-card" style="margin-bottom: 0.5rem; padding: 1rem;">
#                         <div style="display: flex; align-items: start;">
#                             <div style="margin-right: 0.5rem; color: #ff4444;">•</div>
#                             <div>{symptom}</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown("### Management & Control")
                
#                 st.markdown("#### Treatment Recommendations")
#                 for treatment in info['treatment']:
#                     st.markdown(f"""
#                     <div class="history-card" style="margin-bottom: 0.5rem; padding: 1rem; border-left-color: #4CAF50;">
#                         <div style="display: flex; align-items: start;">
#                             <div style="margin-right: 0.5rem; color: #4CAF50;">✓</div>
#                             <div>{treatment}</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 st.markdown("#### Preventive Measures")
#                 st.markdown("""
#                 <div style="background: rgba(33, 150, 243, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
#                     <p style="margin-bottom: 0.5rem; font-weight: 600; color: #2196F3;">Prevention Tips:</p>
#                     <p style="font-size: 0.95rem;">• Regular field monitoring</p>
#                     <p style="font-size: 0.95rem;">• Proper irrigation management</p>
#                     <p style="font-size: 0.95rem;">• Use of certified disease-free seeds</p>
#                     <p style="font-size: 0.95rem;">• Crop rotation practices</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown("---")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_history_page():
#     """Analysis history page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## Analysis History")
    
#     if 'history' not in st.session_state or len(st.session_state.history) == 0:
#         st.markdown("""
#         <div class="empty-state">
#             <h3>No Analysis History</h3>
#             <p>Start by analyzing some rice leaf images to see your history here.</p>
#             <div style="margin-top: 2rem;">
#                 <a href="#disease-detection">
#                     <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
#                             color: white; border: none; padding: 0.8rem 2rem; 
#                             border-radius: 25px; cursor: pointer; font-weight: 600;">
#                         Go to Detection Page
#                     </button>
#                 </a>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         # Sort history by timestamp (newest first)
#         history = sorted(st.session_state.history, 
#                         key=lambda x: x.get('timestamp', ''), 
#                         reverse=True)
        
#         for idx, record in enumerate(history):
#             disease_data = DISEASE_INFO.get(record.get('class', ''), {})
#             severity = disease_data.get('severity', 'UNKNOWN') if disease_data else 'UNKNOWN'
#             severity_class = {
#                 'HIGH': 'severity-high',
#                 'MEDIUM': 'severity-medium',
#                 'LOW-MEDIUM': 'severity-low'
#             }.get(severity, 'severity-medium')
            
#             with st.expander(f"Analysis {idx+1}: {disease_data.get('name', 'Unknown Disease')} - {record.get('timestamp', 'Unknown time')}", 
#                            expanded=idx==0):
#                 col1, col2 = st.columns([1, 2])
                
#                 with col1:
#                     confidence = record.get('confidence', 0) * 100
#                     st.markdown(f"""
#                     <div style="text-align: center; padding: 1rem;">
#                         <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
#                             {confidence:.1f}%
#                         </div>
#                         <div style="color: #666; font-size: 0.9rem;">Confidence</div>
#                         <div style="margin-top: 1rem;">
#                             <span class="severity-badge {severity_class}" style="font-size: 0.8rem;">
#                                 {severity}
#                             </span>
#                         </div>
#                         <div style="margin-top: 1rem; font-size: 0.9rem; color: #888;">
#                             Source: {record.get('source', 'Unknown').title()}
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with col2:
#                     st.markdown(f"**Disease:** {disease_data.get('name', 'Unknown')}")
#                     st.markdown(f"**Scientific Name:** {disease_data.get('scientific', 'Unknown')}")
#                     st.markdown(f"**Detection Time:** {record.get('timestamp', 'Unknown')}")
                    
#                     # Probability distribution
#                     if 'all_probabilities' in record:
#                         st.markdown("**Probability Distribution:**")
#                         prob_df = pd.DataFrame(list(record['all_probabilities'].items()),
#                                              columns=['Disease', 'Probability'])
#                         prob_df['Probability'] = prob_df['Probability'] * 100
#                         st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_about_page():
#     """About page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("## About AgroSight")
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown("""
#         ### Overview
#         **AgroSight** is an advanced AI-powered platform designed to detect rice leaf diseases 
#         using state-of-the-art deep learning technology. The system employs a ResNet-34 
#         convolutional neural network architecture to classify three major rice diseases 
#         with high accuracy and reliability.
        
#         ### Core Features
#         - **Real-time Detection**: Instant disease predictions from uploaded or captured images
#         - **High Accuracy**: Trained on extensive datasets of rice leaf images
#         - **Comprehensive Analysis**: Detailed disease information and treatment recommendations
#         - **User-Friendly Interface**: Accessible to farmers, agronomists, and researchers
#         - **Historical Tracking**: Maintains analysis history for reference and comparison
        
#         ### Technology Architecture
#         """)
        
#         tech_stack = {
#             "Deep Learning Framework": "PyTorch",
#             "Neural Network": "ResNet-34 Architecture",
#             "Web Framework": "Streamlit",
#             "Visualization": "Plotly & Matplotlib",
#             "Image Processing": "OpenCV & PIL",
#             "Data Handling": "Pandas & NumPy"
#         }
        
#         for tech, desc in tech_stack.items():
#             st.markdown(f"""
#             <div class="history-card" style="margin-bottom: 0.5rem;">
#                 <strong>{tech}:</strong> {desc}
#             </div>
#             """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         ### Detected Diseases
        
#         **1. Bacterial Blight**
#         - **Pathogen**: *Xanthomonas oryzae* pv. *oryzae*
#         - **Impact**: Can cause up to 70% yield loss
#         - **Season**: Prevalent in wet seasons
        
#         **2. Brown Spot**
#         - **Pathogen**: *Bipolaris oryzae*
#         - **Impact**: Reduces grain quality and yield
#         - **Season**: Common in nutrient-deficient soils
        
#         **3. Leaf Smut**
#         - **Pathogen**: *Entyloma oryzae*
#         - **Impact**: Affects photosynthesis efficiency
#         - **Season**: Found in humid conditions
        
#         ### Mission & Vision
        
#         **Our Mission:**
#         To empower farmers with accessible, accurate, and timely disease detection tools,
#         supporting sustainable agriculture and strengthening global food security.
        
#         **Our Vision:**
#         To become the leading AI-powered agricultural diagnostic platform,
#         revolutionizing crop disease management worldwide.
        
#         ### Development Team
#         This platform is developed by a team of AI researchers, agricultural scientists,
#         and software engineers dedicated to bridging technology and agriculture.
#         """)
        
#         st.markdown("---")
#         st.markdown("""
#         ### Contact & Support
#         For technical support, feature requests, or collaboration opportunities:
        
#         - **Email**: support@agrosight.ai
#         - **Documentation**: docs.agrosight.ai
#         - **GitHub**: github.com/agrosight
#         - **Research Papers**: research.agrosight.ai
        
#         ### Version Information
#         - **Current Version**: 2.0.1
#         - **Last Updated**: 2025
#         - **Model Version**: ResNet-34 v1.2
#         - **Data Version**: RiceLeaf-2024
#         """)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <p style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">
#             AgroSight - Revolutionizing Agricultural Diagnostics
#         </p>
#         <p style="font-size: 1rem; margin-bottom: 0.5rem; opacity: 0.9;">
#             Advanced AI for Sustainable Agriculture
#         </p>
#         <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem;">
#             <span style="font-size: 0.9rem;">Accuracy: >95%</span>
#             <span style="font-size: 0.9rem;">Response Time: <2s</span>
#             <span style="font-size: 0.9rem;">Diseases: 3+</span>
#         </div>
#         <p style="font-size: 0.85rem; margin-top: 1.5rem; opacity: 0.8;">
#             © 2025 AgroSight Research Lab | Powered by PyTorch & Streamlit
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     # Initialize session state
#     if 'analyzed' not in st.session_state:
#         st.session_state['analyzed'] = False
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []
#     if 'captured_image' not in st.session_state:
#         st.session_state['captured_image'] = None
#     if 'captured_image_display' not in st.session_state:
#         st.session_state['captured_image_display'] = None
#     if 'results' not in st.session_state:
#         st.session_state['results'] = {}
    
#     # Add Streamlit run warning suppression
#     import warnings
#     warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext")
    
#     main()

# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# import json
# import os
# from datetime import datetime
# import cv2
# import io

# # ==================== PAGE CONFIGURATION ====================
# st.set_page_config(
#     page_title="AgroSight - Rice Disease Detection",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==================== CUSTOM CSS ====================
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
#     * {
#         font-family: 'Poppins', sans-serif;
#         margin: 0;
#         padding: 0;
#         box-sizing: border-box;
#     }
    
#     html, body {
#         margin: 0;
#         padding: 0;
#         overflow-x: hidden;
#         background: #f5f7fa;
#     }
    
#     .main {
#         background: linear-gradient(135deg, #0c2461 0%, #1e3799 20%, #4a69bd 100%);
#         min-height: 100vh;
#         padding: 0;
#     }
    
#     .stApp {
#         background: transparent;
#         padding-top: 0;
#     }
    
#     /* Hero Section */
#     .hero-section {
#         background: linear-gradient(135deg, 
#             rgba(26, 188, 156, 0.95) 0%, 
#             rgba(22, 160, 133, 0.95) 30%,
#             rgba(19, 141, 117, 0.95) 100%);
#         padding: 4rem 2rem;
#         border-radius: 0 0 40px 40px;
#         box-shadow: 0 25px 70px rgba(0,0,0,0.3);
#         text-align: center;
#         margin-bottom: 3rem;
#         animation: fadeInDown 1s ease-out;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .hero-section::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         bottom: 0;
#         background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100"><path fill="rgba(255,255,255,0.1)" d="M0,50 Q250,100 500,50 T1000,50 V100 H0 Z"/></svg>');
#         background-size: cover;
#         opacity: 0.2;
#     }
    
#     .hero-title {
#         font-size: 3.8rem;
#         font-weight: 800;
#         color: white;
#         text-shadow: 3px 3px 8px rgba(0,0,0,0.25);
#         margin-bottom: 1rem;
#         letter-spacing: 2px;
#         background: linear-gradient(45deg, #ffffff 30%, #e8f5e9 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         line-height: 1.1;
#     }
    
#     .hero-subtitle {
#         font-size: 1.4rem;
#         color: #f0f7f5;
#         font-weight: 400;
#         margin-top: 0;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
#         max-width: 800px;
#         margin: 0 auto 1.5rem auto;
#         line-height: 1.6;
#         opacity: 0.95;
#     }
    
#     .hero-tagline {
#         font-size: 1.1rem;
#         color: #c8f7ec;
#         font-weight: 300;
#         max-width: 700px;
#         margin: 1rem auto 0 auto;
#         line-height: 1.5;
#         opacity: 0.9;
#     }
    
#     /* Cards */
#     .card {
#         background: rgba(255, 255, 255, 0.98);
#         backdrop-filter: blur(15px);
#         -webkit-backdrop-filter: blur(15px);
#         padding: 2.2rem;
#         border-radius: 22px;
#         box-shadow: 
#             0 12px 30px rgba(0,0,0,0.08),
#             0 3px 12px rgba(0,0,0,0.05),
#             inset 0 1px 0 rgba(255,255,255,0.6);
#         margin-bottom: 1.8rem;
#         transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
#         border: 1px solid rgba(255,255,255,0.8);
#         animation: fadeInUp 0.7s ease-out;
#     }
    
#     .card:hover {
#         transform: translateY(-6px) scale(1.008);
#         box-shadow: 
#             0 20px 45px rgba(0,0,0,0.12),
#             0 10px 20px rgba(0,0,0,0.08),
#             inset 0 1px 0 rgba(255,255,255,0.8);
#     }
    
#     /* Section Titles */
#     .section-title {
#         font-size: 1.8rem;
#         font-weight: 700;
#         color: #2c3e50;
#         margin-bottom: 1.8rem;
#         position: relative;
#         padding-bottom: 0.8rem;
#     }
    
#     .section-title::after {
#         content: '';
#         position: absolute;
#         bottom: 0;
#         left: 0;
#         width: 60px;
#         height: 4px;
#         background: linear-gradient(90deg, #667eea, #764ba2);
#         border-radius: 2px;
#     }
    
#     /* Metric Cards */
#     .metric-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
#         gap: 1.2rem;
#         margin: 2rem 0;
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
#         color: white;
#         padding: 1.8rem 1.2rem;
#         border-radius: 18px;
#         text-align: center;
#         box-shadow: 
#             0 8px 25px rgba(0,0,0,0.15),
#             0 3px 10px rgba(0,0,0,0.1);
#         transition: all 0.3s ease;
#         position: relative;
#         overflow: hidden;
#         border: none;
#     }
    
#     .metric-card::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
#         transition: 0.6s;
#     }
    
#     .metric-card:hover::before {
#         left: 100%;
#     }
    
#     .metric-card:hover {
#         transform: translateY(-4px) scale(1.03);
#         box-shadow: 
#             0 15px 35px rgba(0,0,0,0.2),
#             0 8px 20px rgba(0,0,0,0.15);
#     }
    
#     .metric-value {
#         font-size: 2.6rem;
#         font-weight: 800;
#         margin: 0.8rem 0;
#         text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
#         background: linear-gradient(45deg, #ffffff 30%, #e0e0e0 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         line-height: 1;
#     }
    
#     .metric-label {
#         font-size: 1rem;
#         font-weight: 500;
#         opacity: 0.92;
#         text-transform: uppercase;
#         letter-spacing: 0.8px;
#         margin-top: 0.5rem;
#     }
    
#     /* Prediction Card */
#     .prediction-card {
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#         color: white;
#         padding: 2.2rem;
#         border-radius: 22px;
#         box-shadow: 
#             0 18px 40px rgba(0,0,0,0.18),
#             0 8px 20px rgba(0,0,0,0.12);
#         margin: 1.8rem 0;
#         border: none;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .prediction-card::after {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px);
#         background-size: 25px 25px;
#         opacity: 0.4;
#         animation: moveBackground 25s linear infinite;
#     }
    
#     .prediction-title {
#         font-size: 1.8rem;
#         font-weight: 700;
#         margin-bottom: 1.2rem;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
#         position: relative;
#         z-index: 1;
#         color: white;
#     }
    
#     .prediction-disease {
#         font-size: 3rem;
#         font-weight: 800;
#         text-transform: uppercase;
#         letter-spacing: 1.5px;
#         text-shadow: 
#             2px 2px 5px rgba(0,0,0,0.25),
#             0 0 15px rgba(255,255,255,0.3);
#         margin: 1rem 0;
#         position: relative;
#         z-index: 1;
#         background: linear-gradient(45deg, #ffffff 30%, #e8f5e9 70%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         line-height: 1.1;
#     }
    
#     /* Disease Info */
#     .disease-info {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 1.8rem;
#         border-radius: 18px;
#         margin-top: 1.5rem;
#         box-shadow: 0 12px 28px rgba(0,0,0,0.15);
#         border: none;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .disease-info::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         height: 4px;
#         background: linear-gradient(90deg, 
#             #ff0000, #ff9900, #ffff00, #00ff00, 
#             #00ffff, #0000ff, #9900ff);
#         background-size: 200% 100%;
#         animation: rainbow 4s linear infinite;
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.9rem 2.2rem;
#         font-size: 1.05rem;
#         font-weight: 600;
#         border-radius: 12px;
#         cursor: pointer;
#         transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
#         box-shadow: 
#             0 6px 18px rgba(0,0,0,0.15),
#             0 3px 10px rgba(0,0,0,0.1);
#         position: relative;
#         overflow: hidden;
#         letter-spacing: 0.5px;
#         text-transform: none;
#         min-height: 48px;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         gap: 0.5rem;
#     }
    
#     .stButton > button::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
#         transition: 0.5s;
#     }
    
#     .stButton > button:hover::before {
#         left: 100%;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-3px);
#         box-shadow: 
#             0 12px 30px rgba(0,0,0,0.2),
#             0 8px 20px rgba(0,0,0,0.15);
#         background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
#     }
    
#     /* Sidebar */
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, 
#             rgba(26, 35, 126, 0.95) 0%, 
#             rgba(48, 63, 159, 0.95) 100%) !important;
#         backdrop-filter: blur(15px);
#         -webkit-backdrop-filter: blur(15px);
#         color: white;
#         border-right: 2px solid rgba(255,255,255,0.12);
#         padding: 1.5rem 1rem;
#     }
    
#     /* Upload Section - FIXED DESIGN */
#     .upload-section-container {
#         border: 3px dashed rgba(102, 126, 234, 0.5);
#         border-radius: 20px;
#         padding: 3rem 2rem;
#         text-align: center;
#         background: rgba(102, 126, 234, 0.05);
#         backdrop-filter: blur(8px);
#         -webkit-backdrop-filter: blur(8px);
#         transition: all 0.3s ease;
#         min-height: 320px;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         align-items: center;
#         position: relative;
#         overflow: hidden;
#         margin: 1rem 0;
#     }
    
#     .upload-section-container:hover {
#         border-color: rgba(118, 75, 162, 0.7);
#         background: rgba(118, 75, 162, 0.08);
#         transform: translateY(-3px);
#         box-shadow: 0 15px 35px rgba(0,0,0,0.08);
#     }
    
#     .upload-icon {
#         font-size: 3.5rem;
#         color: rgba(102, 126, 234, 0.7);
#         margin-bottom: 1.2rem;
#         transition: all 0.3s ease;
#     }
    
#     .upload-title {
#         font-size: 1.4rem;
#         font-weight: 600;
#         color: #2c3e50;
#         margin-bottom: 0.8rem;
#     }
    
#     .upload-subtitle {
#         font-size: 0.95rem;
#         color: #666;
#         margin-bottom: 1.5rem;
#         line-height: 1.5;
#         max-width: 400px;
#     }
    
#     .upload-hint {
#         font-size: 0.85rem;
#         color: #888;
#         margin-top: 1.5rem;
#         font-style: italic;
#     }
    
#     /* Camera Section */
#     .camera-section-container {
#         border: 3px dashed rgba(38, 166, 154, 0.5);
#         border-radius: 20px;
#         padding: 2.5rem 1.8rem;
#         text-align: center;
#         background: rgba(38, 166, 154, 0.05);
#         backdrop-filter: blur(8px);
#         -webkit-backdrop-filter: blur(8px);
#         transition: all 0.3s ease;
#         min-height: 320px;
#         position: relative;
#         overflow: hidden;
#         margin: 1rem 0;
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         align-items: center;
#     }
    
#     .camera-section-container:hover {
#         border-color: rgba(38, 166, 154, 0.7);
#         background: rgba(38, 166, 154, 0.1);
#         transform: translateY(-3px);
#         box-shadow: 0 15px 35px rgba(0,0,0,0.08);
#     }
    
#     .camera-icon {
#         font-size: 3.5rem;
#         color: rgba(38, 166, 154, 0.7);
#         margin-bottom: 1.2rem;
#         transition: all 0.3s ease;
#     }
    
#     /* Image Preview */
#     .image-preview-container {
#         border-radius: 16px;
#         overflow: hidden;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.1);
#         margin: 1.5rem 0;
#         border: 2px solid rgba(255,255,255,0.8);
#         background: white;
#     }
    
#     .image-preview-container img {
#         width: 100%;
#         height: auto;
#         display: block;
#         transition: transform 0.3s ease;
#     }
    
#     .image-preview-container img:hover {
#         transform: scale(1.02);
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 1.5rem;
#         padding: 1rem 0;
#         border-bottom: 2px solid rgba(0,0,0,0.05);
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background: rgba(255, 255, 255, 0.9);
#         border-radius: 12px;
#         padding: 0.9rem 1.8rem;
#         font-weight: 600;
#         font-size: 1rem;
#         color: #555;
#         border: 1.5px solid rgba(0,0,0,0.08);
#         transition: all 0.25s ease;
#         margin: 0;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 20px rgba(0,0,0,0.1);
#         background: white;
#         border-color: rgba(102, 126, 234, 0.3);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white !important;
#         border-color: rgba(255,255,255,0.3) !important;
#         box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
#     }
    
#     /* Progress Bar */
#     .confidence-bar {
#         background: rgba(255,255,255,0.15);
#         border-radius: 12px;
#         padding: 0.7rem;
#         margin-top: 1.2rem;
#         position: relative;
#         overflow: hidden;
#         border: 1px solid rgba(255,255,255,0.2);
#     }
    
#     .confidence-fill {
#         position: absolute;
#         top: 0;
#         left: 0;
#         height: 100%;
#         background: linear-gradient(90deg, #4CAF50, #8BC34A);
#         border-radius: 10px;
#         transition: width 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
#         box-shadow: 0 0 15px rgba(76, 175, 80, 0.4);
#     }
    
#     /* Footer */
#     .footer {
#         text-align: center;
#         padding: 2.5rem;
#         background: rgba(0, 0, 0, 0.25);
#         backdrop-filter: blur(15px);
#         border-radius: 22px;
#         margin-top: 3rem;
#         color: white;
#         border: 1px solid rgba(255,255,255,0.08);
#         box-shadow: 0 10px 30px rgba(0,0,0,0.15);
#     }
    
#     /* Animations */
#     @keyframes fadeInDown {
#         from {
#             opacity: 0;
#             transform: translateY(-40px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     @keyframes fadeInUp {
#         from {
#             opacity: 0;
#             transform: translateY(40px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     @keyframes moveBackground {
#         from {
#             transform: rotate(0deg);
#         }
#         to {
#             transform: rotate(360deg);
#         }
#     }
    
#     @keyframes rainbow {
#         0% {
#             background-position: 0% 50%;
#         }
#         100% {
#             background-position: 200% 50%;
#         }
#     }
    
#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 8px;
#         height: 8px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: rgba(255, 255, 255, 0.08);
#         border-radius: 4px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6));
#         border-radius: 4px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8));
#     }
    
#     /* Selection */
#     ::selection {
#         background: rgba(102, 126, 234, 0.4);
#         color: white;
#     }
    
#     /* Loading Spinner */
#     .stSpinner > div {
#         border-color: #667eea transparent transparent transparent !important;
#         width: 30px !important;
#         height: 30px !important;
#     }
    
#     /* Empty State */
#     .empty-state {
#         text-align: center;
#         padding: 3.5rem 2rem;
#         color: #666;
#     }
    
#     .empty-state h3 {
#         font-size: 1.7rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#         color: #444;
#     }
    
#     .empty-state p {
#         font-size: 1.05rem;
#         opacity: 0.7;
#         margin-bottom: 1.5rem;
#         line-height: 1.5;
#     }
    
#     /* Disease Severity Badge */
#     .severity-badge {
#         display: inline-block;
#         padding: 0.5rem 1.2rem;
#         border-radius: 20px;
#         font-weight: 600;
#         font-size: 0.85rem;
#         letter-spacing: 0.5px;
#         margin-left: 0.8rem;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         border: 1px solid rgba(255,255,255,0.2);
#     }
    
#     .severity-high {
#         background: linear-gradient(135deg, #ff416c, #ff4b2b);
#         color: white;
#     }
    
#     .severity-medium {
#         background: linear-gradient(135deg, #f7971e, #ffd200);
#         color: #333;
#     }
    
#     .severity-low {
#         background: linear-gradient(135deg, #11998e, #38ef7d);
#         color: white;
#     }
    
#     /* History Card */
#     .history-card {
#         background: rgba(255, 255, 255, 0.95);
#         border-radius: 14px;
#         padding: 1.3rem;
#         margin-bottom: 0.9rem;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.06);
#         border-left: 4px solid #667eea;
#         transition: all 0.25s ease;
#         border: 1px solid rgba(0,0,0,0.05);
#     }
    
#     .history-card:hover {
#         transform: translateX(4px);
#         box-shadow: 0 6px 20px rgba(0,0,0,0.1);
#         border-left-color: #764ba2;
#     }
    
#     /* Form Elements */
#     .stFileUploader {
#         border: 2px dashed #667eea !important;
#         border-radius: 12px !important;
#         padding: 2rem !important;
#         background: rgba(102, 126, 234, 0.03) !important;
#     }
    
#     .stFileUploader:hover {
#         border-color: #764ba2 !important;
#         background: rgba(118, 75, 162, 0.05) !important;
#     }
    
#     /* Responsive Design */
#     @media (max-width: 768px) {
#         .hero-title {
#             font-size: 2.4rem;
#         }
        
#         .hero-subtitle {
#             font-size: 1.1rem;
#         }
        
#         .card {
#             padding: 1.5rem;
#         }
        
#         .metric-card {
#             padding: 1.4rem 1rem;
#         }
        
#         .metric-value {
#             font-size: 2rem;
#         }
        
#         .prediction-disease {
#             font-size: 2.2rem;
#         }
        
#         .stTabs [data-baseweb="tab"] {
#             padding: 0.7rem 1.2rem;
#             font-size: 0.9rem;
#         }
        
#         .upload-section-container,
#         .camera-section-container {
#             padding: 2rem 1.5rem;
#             min-height: 280px;
#         }
#     }
    
#     /* Tooltips */
#     .tooltip {
#         position: relative;
#         display: inline-block;
#         border-bottom: 1px dotted #667eea;
#     }
    
#     .tooltip .tooltiptext {
#         visibility: hidden;
#         width: 200px;
#         background-color: rgba(0, 0, 0, 0.85);
#         color: #fff;
#         text-align: center;
#         border-radius: 8px;
#         padding: 10px;
#         position: absolute;
#         z-index: 1000;
#         bottom: 125%;
#         left: 50%;
#         margin-left: -100px;
#         opacity: 0;
#         transition: opacity 0.3s;
#         font-size: 0.85rem;
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255,255,255,0.1);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.2);
#     }
    
#     .tooltip:hover .tooltiptext {
#         visibility: visible;
#         opacity: 1;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ==================== MODEL CONFIGURATION ====================
# CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut']
# MODEL_PATH = 'models/best_model.pth'
# IMG_SIZE = 224

# DISEASE_INFO = {
#     'Bacterialblight': {
#         'name': 'Bacterial Blight',
#         'scientific': 'Xanthomonas oryzae pv. oryzae',
#         'symptoms': [
#             'Water-soaked lesions on leaf margins',
#             'Yellow to white lesions with wavy edges',
#             'Systemic infection leads to wilting',
#             'Bacterial ooze visible in morning'
#         ],
#         'treatment': [
#             'Use resistant varieties',
#             'Apply copper-based bactericides',
#             'Maintain field sanitation',
#             'Avoid excessive nitrogen fertilizer'
#         ],
#         'severity': 'HIGH',
#         'color': '#ef5350'
#     },
#     'Brownspot': {
#         'name': 'Brown Spot',
#         'scientific': 'Bipolaris oryzae',
#         'symptoms': [
#             'Circular brown spots with gray centers',
#             'Spots surrounded by yellow halos',
#             'Numerous spots cause leaf blight',
#             'Affects yield and grain quality'
#         ],
#         'treatment': [
#             'Apply fungicides (Mancozeb, Tricyclazole)',
#             'Balance nutrient management',
#             'Ensure proper drainage',
#             'Use disease-free seeds'
#         ],
#         'severity': 'MEDIUM',
#         'color': '#ff9800'
#     },
#     'Leafsmut': {
#         'name': 'Leaf Smut',
#         'scientific': 'Entyloma oryzae',
#         'symptoms': [
#             'Small black angular spots on leaves',
#             'Spots aligned between leaf veins',
#             'Severe infection causes leaf yellowing',
#             'Reduces photosynthesis efficiency'
#         ],
#         'treatment': [
#             'Apply systemic fungicides early',
#             'Remove infected plant debris',
#             'Crop rotation with non-host crops',
#             'Improve air circulation in field'
#         ],
#         'severity': 'LOW-MEDIUM',
#         'color': '#66bb6a'
#     }
# }

# # ==================== MODEL LOADING ====================
# @st.cache_resource
# def load_model():
#     """Load the trained ResNet-34 model"""
#     class ResNet34RiceClassifier(nn.Module):
#         def __init__(self, num_classes=3):
#             super(ResNet34RiceClassifier, self).__init__()
#             self.model = models.resnet34(weights=None)
#             num_features = self.model.fc.in_features
#             self.model.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_features, 512),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(512),
#                 nn.Dropout(0.3),
#                 nn.Linear(512, num_classes)
#             )
        
#         def forward(self, x):
#             return self.model(x)
    
#     model = ResNet34RiceClassifier(num_classes=len(CLASSES))
#     if os.path.exists(MODEL_PATH):
#         try:
#             model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#             model.eval()
#             return model
#         except Exception as e:
#             st.error(f"Error loading model: {e}")
#             return None
#     else:
#         st.error(f"Model file not found at: {MODEL_PATH}")
#         return None

# @st.cache_resource
# def load_metrics():
#     """Load training metrics"""
#     metrics_path = 'results/metrics.json'
#     if os.path.exists(metrics_path):
#         try:
#             with open(metrics_path, 'r') as f:
#                 return json.load(f)
#         except Exception as e:
#             st.warning(f"Error loading metrics: {e}")
#             return None
#     return None

# # ==================== IMAGE PREPROCESSING ====================
# def preprocess_image(image):
#     """Preprocess image for model input"""
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     return transform(image).unsqueeze(0)

# # ==================== CAMERA FUNCTIONS ====================
# def capture_from_camera():
#     """Capture image from webcam"""
#     try:
#         cap = cv2.VideoCapture(0)
        
#         if not cap.isOpened():
#             st.error("⚠️ Cannot access camera. Please check your camera permissions and connections.")
#             return None
        
#         # Set camera resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
#         ret, frame = cap.read()
#         cap.release()
        
#         if ret and frame is not None:
#             # Convert BGR to RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             return Image.fromarray(frame_rgb)
#         else:
#             st.error("Failed to capture image from camera.")
#             return None
#     except Exception as e:
#         st.error(f"Camera error: {str(e)}")
#         return None

# def process_camera_image(image):
#     """Process camera image for display"""
#     if image is None:
#         return None
    
#     try:
#         # Convert PIL Image to numpy array
#         if isinstance(image, Image.Image):
#             img_array = np.array(image)
#         else:
#             img_array = image
        
#         if img_array is None or img_array.size == 0:
#             return image  # Return original if empty
        
#         # Add some image enhancement
#         # Apply CLAHE for contrast enhancement
#         lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         l = clahe.apply(l)
#         lab = cv2.merge((l, a, b))
#         enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
#         return Image.fromarray(np.clip(enhanced, 0, 255).astype('uint8'))
#     except Exception as e:
#         # Return original image if processing fails
#         return image

# # ==================== PREDICTION ====================
# def predict_disease(model, image):
#     """Predict disease from image"""
#     try:
#         with torch.no_grad():
#             image_tensor = preprocess_image(image)
#             output = model(image_tensor)
#             probabilities = torch.nn.functional.softmax(output, dim=1)[0]
#             predicted_class = torch.argmax(probabilities).item()
#             confidence = probabilities[predicted_class].item()
            
#             results = {
#                 'class': CLASSES[predicted_class],
#                 'confidence': confidence,
#                 'all_probabilities': {
#                     CLASSES[i]: float(probabilities[i]) 
#                     for i in range(len(CLASSES))
#                 },
#                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
#             return results
#     except Exception as e:
#         st.error(f"Prediction error: {str(e)}")
#         return None

# # ==================== VISUALIZATION FUNCTIONS ====================
# def create_confidence_gauge(confidence):
#     """Create confidence gauge chart"""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number+delta",
#         value=confidence * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={
#             'text': "Confidence Level",
#             'font': {'size': 20, 'color': 'white', 'family': 'Poppins'}
#         },
#         delta={'reference': 80, 'increasing': {'color': "#38ef7d"}},
#         gauge={
#             'axis': {
#                 'range': [None, 100],
#                 'tickwidth': 1.5,
#                 'tickcolor': "white",
#                 'tickfont': {'color': 'white', 'size': 11}
#             },
#             'bar': {'color': "#1a237e"},
#             'bgcolor': "rgba(255,255,255,0.1)",
#             'borderwidth': 1.5,
#             'bordercolor': "rgba(255,255,255,0.3)",
#             'steps': [
#                 {'range': [0, 50], 'color': 'rgba(239, 83, 80, 0.5)'},
#                 {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.5)'},
#                 {'range': [75, 100], 'color': 'rgba(102, 187, 106, 0.5)'}
#             ],
#             'threshold': {
#                 'line': {'color': "white", 'width': 3},
#                 'thickness': 0.75,
#                 'value': 90
#             }
#         }
#     ))
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': "white", 'family': "Poppins"},
#         height=280,
#         margin=dict(l=40, r=40, t=60, b=40)
#     )
#     return fig

# def create_probability_chart(probabilities):
#     """Create probability bar chart"""
#     df = pd.DataFrame(list(probabilities.items()), 
#                      columns=['Disease', 'Probability'])
#     df['Probability'] = df['Probability'] * 100
#     df['Disease_Name'] = df['Disease'].map(lambda x: DISEASE_INFO[x]['name'])
    
#     fig = px.bar(
#         df, x='Probability', y='Disease_Name',
#         orientation='h',
#         color='Probability',
#         color_continuous_scale='Viridis',
#         labels={'Probability': 'Probability (%)', 'Disease_Name': 'Disease'},
#         title='Prediction Probabilities',
#         text='Probability'
#     )
    
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(255,255,255,0.95)',
#         font={'family': 'Poppins', 'size': 12},
#         title_font={'size': 18, 'color': '#2c3e50', 'family': 'Poppins'},
#         height=350,
#         showlegend=False,
#         margin=dict(l=0, r=0, t=50, b=0)
#     )
    
#     fig.update_traces(
#         marker_line_color='rgb(30, 30, 30)',
#         marker_line_width=1.2,
#         texttemplate='%{x:.1f}%',
#         textposition='outside',
#         textfont={'size': 11, 'color': '#2c3e50'}
#     )
    
#     fig.update_xaxes(range=[0, 100])
    
#     return fig

# def create_disease_severity_chart():
#     """Create disease severity comparison chart"""
#     disease_names = [DISEASE_INFO[disease]['name'] for disease in CLASSES]
#     severities = []
#     colors = []
#     descriptions = []
    
#     for disease in CLASSES:
#         info = DISEASE_INFO[disease]
#         severity_value = {
#             'HIGH': 90,
#             'MEDIUM': 65,
#             'LOW-MEDIUM': 40
#         }.get(info['severity'], 50)
#         severities.append(severity_value)
#         colors.append(info['color'])
#         descriptions.append(info['severity'])
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=disease_names,
#             y=severities,
#             marker_color=colors,
#             text=descriptions,
#             textposition='outside',
#             hovertemplate='<b>%{x}</b><br>Severity: %{text}<br>Level: %{y}%<extra></extra>',
#             marker_line_color='rgba(0,0,0,0.3)',
#             marker_line_width=1.5
#         )
#     ])
    
#     fig.update_layout(
#         title={
#             'text': 'Disease Severity Comparison',
#             'font': {'size': 18, 'color': '#2c3e50', 'family': 'Poppins'}
#         },
#         xaxis_title={
#             'text': 'Disease',
#             'font': {'size': 13, 'color': '#555'}
#         },
#         yaxis_title={
#             'text': 'Severity Level (%)',
#             'font': {'size': 13, 'color': '#555'}
#         },
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(255,255,255,0.95)',
#         font={'family': 'Poppins', 'size': 12},
#         height=350,
#         showlegend=False,
#         margin=dict(l=0, r=0, t=50, b=0)
#     )
    
#     fig.update_yaxes(range=[0, 100])
    
#     return fig

# # ==================== MAIN APP ====================
# def main():
#     # Hero Section
#     st.markdown("""
#     <div class="hero-section">
#         <h1 class="hero-title">🌾 AgroSight</h1>
#         <p class="hero-subtitle">Advanced AI-Powered Rice Disease Detection System</p>
#         <p class="hero-tagline">
#             Empower your agricultural decisions with cutting-edge deep learning technology. 
#             Detect, analyze, and manage rice leaf diseases with precision and confidence.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load model
#     with st.spinner("Loading AI model..."):
#         model = load_model()
    
#     metrics = load_metrics()
    
#     if model is None:
#         st.error("""
#         ## Model Not Found
#         The trained model file was not found. Please ensure:
#         1. You have trained the model using `train_model.py`
#         2. The model file exists at: `models/best_model.pth`
#         3. You have the required dependencies installed
#         """)
#         return
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("""
#         <div style="text-align: center; margin-bottom: 2rem;">
#             <div style="font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 0.5rem;">
#                 Navigation
#             </div>
#             <div style="height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); 
#                  border-radius: 2px; margin: 0 auto; width: 60px;"></div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         page = st.radio(
#             "Select Page",
#             ["🔍 Disease Detection", "📊 Model Performance", "📚 Disease Information", 
#              "📋 Analysis History", "ℹ️ About"],
#             label_visibility="collapsed"
#         )
        
#         # Extract page name from selection
#         page_name = page.split(" ")[-1]
        
#         st.markdown("---")
#         st.markdown("### System Status")
        
#         # Status indicators
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown(f"""
#             <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
#                 <div class="metric-label">Model Status</div>
#                 <div class="metric-value" style="font-size: 1.4rem; color: #38ef7d;">✓ Active</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown(f"""
#             <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
#                 <div class="metric-label">Diseases</div>
#                 <div class="metric-value" style="font-size: 1.4rem;">{len(CLASSES)}</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         if metrics:
#             st.markdown("### Performance Metrics")
#             col1, col2 = st.columns(2)
#             with col1:
#                 accuracy = metrics.get('test_accuracy', 0) * 100
#                 st.metric("Accuracy", f"{accuracy:.1f}%")
#             with col2:
#                 if 'classification_report' in metrics:
#                     avg_f1 = np.mean([metrics['classification_report'][cls]['f1-score'] for cls in CLASSES]) * 100
#                 else:
#                     avg_f1 = 0
#                 st.metric("Avg F1-Score", f"{avg_f1:.1f}%")
        
#         st.markdown("---")
#         st.markdown("### Quick Actions")
#         if st.button("🗑️ Clear History", key="clear_history", use_container_width=True):
#             if 'history' in st.session_state:
#                 st.session_state.history = []
#                 st.success("History cleared successfully!", icon="✅")
        
#         st.markdown("---")
#         st.markdown("""
#         <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); 
#              border-radius: 12px; margin-top: 1rem;'>
#             <p style='color: white; font-size: 0.9rem; margin-bottom: 0.3rem; font-weight: 500;'>
#                 System Version 2.1.0
#             </p>
#             <p style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>
#                 Powered by ResNet-34 AI
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Main Content Routing
#     if page_name == "Detection":
#         show_detection_page(model)
#     elif page_name == "Performance":
#         show_performance_page(metrics)
#     elif page_name == "Information":
#         show_disease_info_page()
#     elif page_name == "History":
#         show_history_page()
#     else:
#         show_about_page()

# def show_detection_page(model):
#     """Disease detection page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">Disease Detection</h2>', unsafe_allow_html=True)
#     st.markdown('<p style="color: #666; margin-bottom: 1.5rem; font-size: 1.05rem;">Upload or capture an image of a rice leaf for AI-powered disease analysis.</p>', unsafe_allow_html=True)
    
#     # Create tabs for different input methods
#     tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Capture from Camera"])
    
#     with tab1:
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             # Custom upload section
#             st.markdown("""
#             <div class="upload-section-container">
#                 <div class="upload-icon">📤</div>
#                 <div class="upload-title">Upload Rice Leaf Image</div>
#                 <div class="upload-subtitle">
#                     Drag and drop your image here or click to browse.<br>
#                     Supported formats: JPG, JPEG, PNG
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             uploaded_file = st.file_uploader(
#                 "Choose an image file",
#                 type=['jpg', 'jpeg', 'png'],
#                 help="Select a clear image of a rice leaf for accurate analysis",
#                 label_visibility="collapsed"
#             )
            
#             if uploaded_file is not None:
#                 try:
#                     image = Image.open(uploaded_file).convert('RGB')
#                     st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
#                     st.image(image, caption='📸 Uploaded Image Preview', use_container_width=True)
#                     st.markdown('</div>', unsafe_allow_html=True)
                    
#                     if st.button("🔍 Analyze Uploaded Image", key="analyze_upload", use_container_width=True):
#                         with st.spinner("🤖 AI is analyzing the image..."):
#                             results = predict_disease(model, image)
                            
#                             if results:
#                                 # Add to history
#                                 if 'history' not in st.session_state:
#                                     st.session_state.history = []
#                                 st.session_state.history.append({
#                                     **results,
#                                     'source': 'upload',
#                                     'filename': uploaded_file.name if hasattr(uploaded_file, 'name') else 'Uploaded Image',
#                                     'image': image
#                                 })
                                
#                                 st.session_state['results'] = results
#                                 st.session_state['analyzed'] = True
#                                 st.session_state['image_source'] = 'upload'
#                                 st.rerun()
#                             else:
#                                 st.error("Failed to analyze image. Please try again.")
#                 except Exception as e:
#                     st.error(f"Error loading image: {str(e)}")
        
#         with col2:
#             if 'analyzed' in st.session_state and st.session_state.get('analyzed') and st.session_state.get('image_source') == 'upload':
#                 results = st.session_state.get('results', {})
#                 if results:
#                     disease = results.get('class', 'Unknown')
#                     confidence = results.get('confidence', 0)
#                     disease_name = DISEASE_INFO.get(disease, {}).get('name', disease)
                    
#                     st.markdown(f"""
#                     <div class="prediction-card">
#                         <div class="prediction-title">🎯 Detection Result</div>
#                         <div class="prediction-disease">{disease_name}</div>
#                         <div style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.95;">
#                             Confidence Level: <strong>{confidence*100:.2f}%</strong>
#                         </div>
#                         <div style="margin-top: 0.5rem; font-size: 0.95rem; opacity: 0.85;">
#                             Detection Time: {results.get('timestamp', 'N/A')}
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.plotly_chart(
#                         create_confidence_gauge(confidence),
#                         use_container_width=True,
#                         config={'displayModeBar': False}
#                     )
    
#     with tab2:
#         st.markdown("""
#         <div class="camera-section-container">
#             <div class="camera-icon">📷</div>
#             <div class="upload-title">Real-time Camera Capture</div>
#             <div class="upload-subtitle">
#                 Capture a photo of rice leaf using your webcam.<br>
#                 Ensure good lighting and clear focus for best results.
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             if st.button("📸 Capture Image", key="capture", use_container_width=True):
#                 with st.spinner("Accessing camera..."):
#                     captured_image = capture_from_camera()
#                     if captured_image:
#                         st.session_state['captured_image'] = captured_image
#                         st.session_state['captured_image_display'] = captured_image
#                         st.success("✅ Image captured successfully!")
#                     else:
#                         st.error("Failed to capture image. Please check your camera.")
        
#         with col2:
#             if 'captured_image_display' in st.session_state and st.session_state['captured_image_display']:
#                 try:
#                     processed_image = process_camera_image(st.session_state['captured_image_display'])
#                     if processed_image:
#                         st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
#                         st.image(processed_image, caption='📸 Captured Image Preview', use_container_width=True)
#                         st.markdown('</div>', unsafe_allow_html=True)
                        
#                         if st.button("🔍 Analyze Captured Image", key="analyze_capture", use_container_width=True):
#                             if 'captured_image' in st.session_state:
#                                 with st.spinner("🤖 AI is analyzing the captured image..."):
#                                     results = predict_disease(model, st.session_state['captured_image'])
                                    
#                                     if results:
#                                         # Add to history
#                                         if 'history' not in st.session_state:
#                                             st.session_state.history = []
#                                         st.session_state.history.append({
#                                             **results,
#                                             'source': 'camera',
#                                             'filename': 'Camera Capture',
#                                             'image': st.session_state['captured_image']
#                                         })
                                        
#                                         st.session_state['results'] = results
#                                         st.session_state['analyzed'] = True
#                                         st.session_state['image_source'] = 'camera'
#                                         st.rerun()
#                                     else:
#                                         st.error("Failed to analyze captured image. Please try again.")
#                 except Exception as e:
#                     st.error(f"Error displaying image: {str(e)}")
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Show detailed results
#     if 'analyzed' in st.session_state and st.session_state.get('analyzed'):
#         results = st.session_state.get('results', {})
#         if results:
#             disease = results.get('class', '')
#             disease_info = DISEASE_INFO.get(disease, {})
            
#             st.markdown('<div class="card">', unsafe_allow_html=True)
#             st.markdown('<h2 class="section-title">📋 Detailed Analysis Report</h2>', unsafe_allow_html=True)
            
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 if 'all_probabilities' in results:
#                     st.plotly_chart(
#                         create_probability_chart(results['all_probabilities']),
#                         use_container_width=True,
#                         config={'displayModeBar': False}
#                     )
                
#                 if disease_info:
#                     severity = disease_info.get('severity', 'UNKNOWN')
#                     severity_class = {
#                         'HIGH': 'severity-high',
#                         'MEDIUM': 'severity-medium',
#                         'LOW-MEDIUM': 'severity-low'
#                     }.get(severity, 'severity-medium')
                    
#                     st.markdown("### ⚠️ Disease Severity Assessment")
#                     st.markdown(f"""
#                     <div style="display: flex; align-items: center; margin: 1rem 0; padding: 1.2rem; 
#                          background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,245,245,0.9)); 
#                          border-radius: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
#                         <span style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-right: 1rem;">
#                             Severity Level:
#                         </span>
#                         <span class="severity-badge {severity_class}">
#                             {severity}
#                         </span>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 if disease_info:
#                     st.markdown(f"""
#                     <div class="disease-info">
#                         <h3 style="margin-bottom: 1rem;">{disease_info.get('name', 'Unknown Disease')}</h3>
#                         <p><strong>🔬 Scientific Name:</strong> {disease_info.get('scientific', 'Unknown')}</p>
#                         <p><strong>📈 Confidence Score:</strong> {results.get('confidence', 0)*100:.2f}%</p>
#                         <p><strong>🕒 Detection Time:</strong> {results.get('timestamp', 'N/A')}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("### 🔍 Key Symptoms")
#                     for symptom in disease_info.get('symptoms', []):
#                         st.markdown(f"""
#                         <div class="history-card" style="padding: 1rem; margin-bottom: 0.7rem;">
#                             <div style="display: flex; align-items: start;">
#                                 <div style="margin-right: 0.7rem; color: #ff4444; font-size: 1.2rem;">•</div>
#                                 <div style="font-size: 0.98rem; color: #444;">{symptom}</div>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     st.markdown("### 💡 Recommended Treatment")
#                     for treatment in disease_info.get('treatment', []):
#                         st.markdown(f"""
#                         <div class="history-card" style="padding: 1rem; margin-bottom: 0.7rem; border-left-color: #4CAF50;">
#                             <div style="display: flex; align-items: start;">
#                                 <div style="margin-right: 0.7rem; color: #4CAF50; font-size: 1.2rem;">✓</div>
#                                 <div style="font-size: 0.98rem; color: #444;">{treatment}</div>
#                             </div>
#                         </div>
#                         """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)

# def show_performance_page(metrics):
#     """Model performance page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">📊 Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
#     if metrics is None:
#         st.info("""
#         ### 📝 No Performance Data Available
#         Model performance metrics are not currently available. 
#         Please train the model first to see detailed performance analytics.
#         """)
#         st.markdown('</div>', unsafe_allow_html=True)
#         return
    
#     # Overall Metrics
#     st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
#     accuracy = metrics.get('test_accuracy', 0) * 100
#     if 'classification_report' in metrics:
#         avg_precision = np.mean([metrics['classification_report'][cls]['precision'] for cls in CLASSES]) * 100
#         avg_recall = np.mean([metrics['classification_report'][cls]['recall'] for cls in CLASSES]) * 100
#         avg_f1 = np.mean([metrics['classification_report'][cls]['f1-score'] for cls in CLASSES]) * 100
#     else:
#         avg_precision = avg_recall = avg_f1 = 0
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Model Accuracy</div>
#             <div class="metric-value">{accuracy:.1f}%</div>
#             <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
#                 Overall prediction accuracy
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Precision</div>
#             <div class="metric-value">{avg_precision:.1f}%</div>
#             <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
#                 Correct positive predictions
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">Recall</div>
#             <div class="metric-value">{avg_recall:.1f}%</div>
#             <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
#                 True positives identified
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         st.markdown(f"""
#         <div class="metric-card">
#             <div class="metric-label">F1-Score</div>
#             <div class="metric-value">{avg_f1:.1f}%</div>
#             <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
#                 Balance of precision & recall
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Classification Report
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">📈 Per-Class Performance Analysis</h2>', unsafe_allow_html=True)
    
#     if 'classification_report' in metrics:
#         class_metrics = []
#         for cls in CLASSES:
#             report = metrics['classification_report'][cls]
#             class_metrics.append({
#                 'Disease': DISEASE_INFO[cls]['name'],
#                 'Precision': f"{report['precision']*100:.2f}%",
#                 'Recall': f"{report['recall']*100:.2f}%",
#                 'F1-Score': f"{report['f1-score']*100:.2f}%",
#                 'Support': report['support']
#             })
        
#         df = pd.DataFrame(class_metrics)
        
#         # Apply custom styling
#         def color_cells(val):
#             if isinstance(val, str) and '%' in val:
#                 num = float(val.replace('%', ''))
#                 if num >= 90:
#                     return 'background-color: rgba(102, 187, 106, 0.2); color: #2e7d32; font-weight: 600;'
#                 elif num >= 80:
#                     return 'background-color: rgba(255, 193, 7, 0.2); color: #f57f17; font-weight: 500;'
#                 else:
#                     return 'background-color: rgba(239, 83, 80, 0.2); color: #c62828; font-weight: 500;'
#             return ''
        
#         styled_df = df.style.applymap(color_cells, subset=['Precision', 'Recall', 'F1-Score'])
#         st.dataframe(
#             styled_df,
#             use_container_width=True,
#             height=250,
#             column_config={
#                 "Disease": st.column_config.TextColumn("Disease", width="medium"),
#                 "Precision": st.column_config.TextColumn("Precision", width="small"),
#                 "Recall": st.column_config.TextColumn("Recall", width="small"),
#                 "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
#                 "Support": st.column_config.NumberColumn("Samples", width="small")
#             }
#         )
#     else:
#         st.warning("Classification report data not available.")
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Visualizations
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">📊 Performance Visualizations</h2>', unsafe_allow_html=True)
    
#     cols = st.columns(2)
    
#     with cols[0]:
#         if 'classification_report' in metrics:
#             fig1 = go.Figure()
            
#             diseases = [DISEASE_INFO[cls]['name'] for cls in CLASSES]
#             precision_vals = [metrics['classification_report'][cls]['precision']*100 for cls in CLASSES]
#             recall_vals = [metrics['classification_report'][cls]['recall']*100 for cls in CLASSES]
            
#             fig1.add_trace(go.Bar(
#                 name='Precision',
#                 x=diseases,
#                 y=precision_vals,
#                 marker_color='#667eea',
#                 text=[f"{v:.1f}%" for v in precision_vals],
#                 textposition='outside'
#             ))
            
#             fig1.add_trace(go.Bar(
#                 name='Recall',
#                 x=diseases,
#                 y=recall_vals,
#                 marker_color='#764ba2',
#                 text=[f"{v:.1f}%" for v in recall_vals],
#                 textposition='outside'
#             ))
            
#             fig1.update_layout(
#                 title={
#                     'text': 'Precision & Recall by Disease',
#                     'font': {'size': 16, 'color': '#2c3e50'}
#                 },
#                 barmode='group',
#                 height=350,
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(255,255,255,0.95)',
#                 font={'family': 'Poppins', 'size': 12},
#                 legend=dict(
#                     orientation="h",
#                     yanchor="bottom",
#                     y=1.02,
#                     xanchor="right",
#                     x=1
#                 ),
#                 margin=dict(l=0, r=0, t=50, b=0)
#             )
            
#             fig1.update_yaxes(range=[0, 100], title_text="Score (%)")
#             st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
    
#     with cols[1]:
#         st.plotly_chart(
#             create_disease_severity_chart(),
#             use_container_width=True,
#             config={'displayModeBar': False}
#         )
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_disease_info_page():
#     """Disease information page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">📚 Rice Disease Information Library</h2>', unsafe_allow_html=True)
#     st.markdown('<p style="color: #666; margin-bottom: 2rem; font-size: 1.05rem;">Comprehensive information about common rice diseases, their symptoms, and treatment methods.</p>', unsafe_allow_html=True)
    
#     for disease, info in DISEASE_INFO.items():
#         with st.expander(f"**{info['name']}** - *{info['scientific']}*", expanded=False):
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 st.markdown("### Disease Characteristics")
                
#                 severity_class = {
#                     'HIGH': 'severity-high',
#                     'MEDIUM': 'severity-medium',
#                     'LOW-MEDIUM': 'severity-low'
#                 }.get(info['severity'], 'severity-medium')
                
#                 st.markdown(f"""
#                 <div style="margin-bottom: 1.5rem; padding: 1.2rem; background: linear-gradient(135deg, 
#                     rgba(255,255,255,0.9), rgba(245,245,245,0.9)); border-radius: 14px; 
#                     box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
#                     <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
#                         <span style="font-weight: 600; margin-right: 0.8rem; color: #2c3e50;">Severity:</span>
#                         <span class="severity-badge {severity_class}">{info['severity']}</span>
#                     </div>
#                     <div style="color: #555; font-size: 0.95rem; line-height: 1.5;">
#                         This disease is classified as <strong>{info['severity']}</strong> severity, 
#                         requiring appropriate management strategies.
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("#### 🔍 Common Symptoms")
#                 for symptom in info['symptoms']:
#                     st.markdown(f"""
#                     <div class="history-card" style="margin-bottom: 0.7rem; padding: 1rem;">
#                         <div style="display: flex; align-items: start;">
#                             <div style="margin-right: 0.7rem; color: #ff4444; font-size: 1.1rem;">•</div>
#                             <div style="font-size: 0.98rem; color: #444;">{symptom}</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown("### 🛡️ Management & Control")
                
#                 st.markdown("#### 💊 Treatment Recommendations")
#                 for treatment in info['treatment']:
#                     st.markdown(f"""
#                     <div class="history-card" style="margin-bottom: 0.7rem; padding: 1rem; border-left-color: #4CAF50;">
#                         <div style="display: flex; align-items: start;">
#                             <div style="margin-right: 0.7rem; color: #4CAF50; font-size: 1.1rem;">✓</div>
#                             <div style="font-size: 0.98rem; color: #444;">{treatment}</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 st.markdown("#### 🚫 Preventive Measures")
#                 st.markdown("""
#                 <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(66, 165, 245, 0.05)); 
#                      padding: 1.2rem; border-radius: 12px; margin-top: 1rem; border: 1px solid rgba(33, 150, 243, 0.2);">
#                     <p style="margin-bottom: 0.8rem; font-weight: 600; color: #2196F3; font-size: 1.05rem;">
#                         🌱 General Prevention Tips:
#                     </p>
#                     <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
#                         • Regular field monitoring and inspection
#                     </p>
#                     <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
#                         • Proper irrigation and water management
#                     </p>
#                     <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
#                         • Use of certified disease-free seeds
#                     </p>
#                     <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
#                         • Crop rotation and field sanitation practices
#                     </p>
#                     <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
#                         • Balanced fertilizer application
#                     </p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown("---")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_history_page():
#     """Analysis history page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">📋 Analysis History</h2>', unsafe_allow_html=True)
    
#     if 'history' not in st.session_state or len(st.session_state.history) == 0:
#         st.markdown("""
#         <div class="empty-state">
#             <h3>📭 No Analysis History Found</h3>
#             <p>You haven't analyzed any images yet. Start by uploading or capturing a rice leaf image to see your history here.</p>
#             <div style="margin-top: 2rem;">
#                 <a href="#disease-detection">
#                     <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
#                             color: white; border: none; padding: 0.9rem 2.2rem; 
#                             border-radius: 12px; cursor: pointer; font-weight: 600;
#                             font-size: 1rem; transition: all 0.3s ease;">
#                         🔍 Go to Detection Page
#                     </button>
#                 </a>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         # Sort history by timestamp (newest first)
#         history = sorted(st.session_state.history, 
#                         key=lambda x: x.get('timestamp', ''), 
#                         reverse=True)
        
#         st.markdown(f"<p style='color: #666; margin-bottom: 1.5rem;'>Total analyses: <strong>{len(history)}</strong></p>", unsafe_allow_html=True)
        
#         for idx, record in enumerate(history):
#             disease = record.get('class', '')
#             disease_info = DISEASE_INFO.get(disease, {})
#             disease_name = disease_info.get('name', 'Unknown Disease')
#             severity = disease_info.get('severity', 'UNKNOWN')
            
#             severity_class = {
#                 'HIGH': 'severity-high',
#                 'MEDIUM': 'severity-medium',
#                 'LOW-MEDIUM': 'severity-low'
#             }.get(severity, 'severity-medium')
            
#             with st.expander(f"**Analysis #{idx+1}:** {disease_name} - {record.get('timestamp', 'Unknown time')}", 
#                            expanded=idx < 2):  # Expand first 2 by default
#                 col1, col2 = st.columns([1, 2])
                
#                 with col1:
#                     confidence = record.get('confidence', 0) * 100
#                     st.markdown(f"""
#                     <div style="text-align: center; padding: 1.2rem; background: linear-gradient(135deg, 
#                          rgba(255,255,255,0.9), rgba(245,245,245,0.9)); border-radius: 14px; 
#                          box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
#                         <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.3rem; 
#                              background: linear-gradient(45deg, #667eea, #764ba2); 
#                              -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
#                              background-clip: text;">
#                             {confidence:.1f}%
#                         </div>
#                         <div style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Confidence</div>
                        
#                         <div style="margin-top: 1rem;">
#                             <span class="severity-badge {severity_class}" style="font-size: 0.8rem;">
#                                 {severity}
#                             </span>
#                         </div>
                        
#                         <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1);">
#                             <div style="font-size: 0.85rem; color: #888;">
#                                 Source: <strong>{record.get('source', 'Unknown').title()}</strong>
#                             </div>
#                             <div style="font-size: 0.8rem; color: #aaa; margin-top: 0.3rem;">
#                                 {record.get('filename', 'Unknown file')}
#                             </div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with col2:
#                     st.markdown(f"**🌾 Disease:** {disease_name}")
#                     if disease_info.get('scientific'):
#                         st.markdown(f"**🔬 Scientific Name:** {disease_info['scientific']}")
#                     st.markdown(f"**🕒 Detection Time:** {record.get('timestamp', 'Unknown')}")
                    
#                     if 'all_probabilities' in record:
#                         st.markdown("**📊 Probability Distribution:**")
#                         prob_df = pd.DataFrame(list(record['all_probabilities'].items()),
#                                              columns=['Disease', 'Probability'])
#                         prob_df['Probability'] = prob_df['Probability'] * 100
#                         prob_df['Disease_Name'] = prob_df['Disease'].map(lambda x: DISEASE_INFO.get(x, {}).get('name', x))
                        
#                         # Create a mini bar chart
#                         fig = px.bar(
#                             prob_df.sort_values('Probability', ascending=True),
#                             x='Probability',
#                             y='Disease_Name',
#                             orientation='h',
#                             color='Probability',
#                             color_continuous_scale='Viridis',
#                             range_x=[0, 100],
#                             height=200
#                         )
                        
#                         fig.update_layout(
#                             paper_bgcolor='rgba(0,0,0,0)',
#                             plot_bgcolor='rgba(255,255,255,0.9)',
#                             font={'family': 'Poppins', 'size': 10},
#                             margin=dict(l=0, r=0, t=0, b=0),
#                             showlegend=False,
#                             xaxis_showgrid=False,
#                             yaxis_showgrid=False
#                         )
                        
#                         fig.update_traces(
#                             marker_line_width=0.5,
#                             texttemplate='%{x:.1f}%',
#                             textposition='outside',
#                             textfont={'size': 9}
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def show_about_page():
#     """About page"""
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">ℹ️ About AgroSight</h2>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown("""
#         ### 🎯 Overview
#         **AgroSight** is an advanced AI-powered platform designed to revolutionize rice disease detection 
#         using state-of-the-art deep learning technology. Our system employs a sophisticated ResNet-34 
#         convolutional neural network architecture to accurately classify three major rice diseases with 
#         exceptional precision and reliability.
        
#         ### ✨ Core Features
#         - **🤖 Real-time AI Detection**: Instant disease predictions from uploaded or captured images
#         - **🎯 High Accuracy**: Trained on extensive datasets of rice leaf images for reliable results
#         - **📋 Comprehensive Analysis**: Detailed disease information and evidence-based treatment recommendations
#         - **👨‍🌾 User-Friendly Interface**: Accessible to farmers, agronomists, researchers, and enthusiasts
#         - **📊 Historical Tracking**: Maintains detailed analysis history for monitoring and comparison
        
#         ### 🏗️ Technology Architecture
#         """)
        
#         tech_stack = {
#             "🧠 Deep Learning Framework": "PyTorch",
#             "🕸️ Neural Network": "ResNet-34 Architecture",
#             "🌐 Web Framework": "Streamlit",
#             "📈 Visualization": "Plotly & Matplotlib",
#             "🖼️ Image Processing": "OpenCV & PIL",
#             "📊 Data Handling": "Pandas & NumPy"
#         }
        
#         for tech, desc in tech_stack.items():
#             st.markdown(f"""
#             <div class="history-card" style="margin-bottom: 0.6rem;">
#                 <div style="display: flex; align-items: center;">
#                     <div style="margin-right: 0.7rem; font-size: 1.2rem;">{tech.split()[0]}</div>
#                     <div>
#                         <strong>{tech.split()[1:]}</strong>: {desc}
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         ### 🌾 Detected Diseases
        
#         **1. 🦠 Bacterial Blight**
#         - **Pathogen**: *Xanthomonas oryzae* pv. *oryzae*
#         - **Impact**: Can cause up to 70% yield loss in severe cases
#         - **Season**: Prevalent in wet seasons with high humidity
        
#         **2. 🍂 Brown Spot**
#         - **Pathogen**: *Bipolaris oryzae*
#         - **Impact**: Reduces grain quality and overall yield significantly
#         - **Season**: Common in nutrient-deficient soils and stressful conditions
        
#         **3. 🍃 Leaf Smut**
#         - **Pathogen**: *Entyloma oryzae*
#         - **Impact**: Affects photosynthesis efficiency and plant vitality
#         - **Season**: Found in humid conditions with poor air circulation
        
#         ### 🎯 Mission & Vision
        
#         **Our Mission:**
#         To empower farmers and agricultural professionals with accessible, accurate, 
#         and timely disease detection tools, supporting sustainable agriculture practices 
#         and strengthening global food security through technological innovation.
        
#         **Our Vision:**
#         To become the world's leading AI-powered agricultural diagnostic platform, 
#         revolutionizing crop disease management and contributing to food sustainability worldwide.
        
#         ### 👥 Development Team
#         This innovative platform is developed by a multidisciplinary team of AI researchers, 
#         agricultural scientists, and software engineers dedicated to bridging the gap between 
#         advanced technology and practical agricultural needs.
#         """)
        
#         st.markdown("---")
#         st.markdown("""
#         ### 📞 Contact & Support
        
#         For technical support, feature requests, or collaboration opportunities:
        
#         - **📧 Email**: support@agrosight.ai
#         - **📚 Documentation**: docs.agrosight.ai
#         - **💻 GitHub**: github.com/agrosight
#         - **📄 Research Papers**: research.agrosight.ai
        
#         ### 🔧 Version Information
#         - **Current Version**: 2.1.0
#         - **Last Updated**: 2025
#         - **Model Version**: ResNet-34 v1.2
#         - **Data Version**: RiceLeaf-2024
#         - **Release Status**: Production Ready
#         """)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <p style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.8rem; 
#            background: linear-gradient(45deg, #ffffff, #e8f5e9); 
#            -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
#            background-clip: text;">
#            🌾 AgroSight - Revolutionizing Agricultural Diagnostics
#         </p>
#         <p style="font-size: 1rem; margin-bottom: 0.8rem; opacity: 0.9; color: rgba(255,255,255,0.9);">
#            Advanced AI for Sustainable Agriculture & Food Security
#         </p>
#         <div style="margin-top: 1.8rem; display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap;">
#             <div style="text-align: center;">
#                 <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">>95%</div>
#                 <div style="font-size: 0.85rem; opacity: 0.8;">Accuracy</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;"><2s</div>
#                 <div style="font-size: 0.85rem; opacity: 0.8;">Response Time</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">3+</div>
#                 <div style="font-size: 0.85rem; opacity: 0.8;">Diseases</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">24/7</div>
#                 <div style="font-size: 0.85rem; opacity: 0.8;">Availability</div>
#             </div>
#         </div>
#         <p style="font-size: 0.85rem; margin-top: 2rem; opacity: 0.7; color: rgba(255,255,255,0.7);">
#             © 2025 AgroSight Research Lab | Powered by PyTorch & Streamlit | All Rights Reserved
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     # Initialize session state
#     if 'analyzed' not in st.session_state:
#         st.session_state['analyzed'] = False
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []
#     if 'captured_image' not in st.session_state:
#         st.session_state['captured_image'] = None
#     if 'captured_image_display' not in st.session_state:
#         st.session_state['captured_image_display'] = None
#     if 'results' not in st.session_state:
#         st.session_state['results'] = {}
#     if 'image_source' not in st.session_state:
#         st.session_state['image_source'] = None
    
#     # Suppress warnings
#     import warnings
#     warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext")
#     warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
#     warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    
#     main()

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import cv2

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="AgroSight - Rice Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        background: #f5f7fa;
    }
    
    .main {
        background: linear-gradient(135deg, #0c2461 0%, #1e3799 20%, #4a69bd 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .stApp {
        background: transparent;
        padding-top: 0;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, 
            rgba(26, 188, 156, 0.95) 0%, 
            rgba(22, 160, 133, 0.95) 30%,
            rgba(19, 141, 117, 0.95) 100%);
        padding: 4rem 2rem;
        border-radius: 0 0 40px 40px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100"><path fill="rgba(255,255,255,0.1)" d="M0,50 Q250,100 500,50 T1000,50 V100 H0 Z"/></svg>');
        background-size: cover;
        opacity: 0.2;
    }
    
    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        color: white;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        letter-spacing: 2px;
        background: linear-gradient(45deg, #ffffff 30%, #e8f5e9 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #f0f7f5;
        font-weight: 400;
        margin-top: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        max-width: 800px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    .hero-tagline {
        font-size: 1.1rem;
        color: #c8f7ec;
        font-weight: 300;
        max-width: 700px;
        margin: 1rem auto 0 auto;
        line-height: 1.5;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 2.2rem;
        border-radius: 22px;
        box-shadow: 
            0 12px 30px rgba(0,0,0,0.08),
            0 3px 12px rgba(0,0,0,0.05),
            inset 0 1px 0 rgba(255,255,255,0.6);
        margin-bottom: 1.8rem;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(255,255,255,0.8);
        animation: fadeInUp 0.7s ease-out;
    }
    
    .card:hover {
        transform: translateY(-6px) scale(1.008);
        box-shadow: 
            0 20px 45px rgba(0,0,0,0.12),
            0 10px 20px rgba(0,0,0,0.08),
            inset 0 1px 0 rgba(255,255,255,0.8);
    }
    
    /* Section Titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1.8rem;
        position: relative;
        padding-bottom: 0.8rem;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.2rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1.8rem 1.2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 
            0 8px 25px rgba(0,0,0,0.15),
            0 3px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        border: none;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        transition: 0.6s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 
            0 15px 35px rgba(0,0,0,0.2),
            0 8px 20px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.8rem 0;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #ffffff 30%, #e0e0e0 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.92;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.5rem;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 22px;
        box-shadow: 
            0 18px 40px rgba(0,0,0,0.18),
            0 8px 20px rgba(0,0,0,0.12);
        margin: 1.8rem 0;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 25px 25px;
        opacity: 0.4;
        animation: moveBackground 25s linear infinite;
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        position: relative;
        z-index: 1;
        color: white;
    }
    
    .prediction-disease {
        font-size: 3rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 
            2px 2px 5px rgba(0,0,0,0.25),
            0 0 15px rgba(255,255,255,0.3);
        margin: 1rem 0;
        position: relative;
        z-index: 1;
        background: linear-gradient(45deg, #ffffff 30%, #e8f5e9 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    
    /* Disease Info */
    .disease-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 18px;
        margin-top: 1.5rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .disease-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            #ff0000, #ff9900, #ffff00, #00ff00, 
            #00ffff, #0000ff, #9900ff);
        background-size: 200% 100%;
        animation: rainbow 4s linear infinite;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.2rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 
            0 6px 18px rgba(0,0,0,0.15),
            0 3px 10px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
        text-transform: none;
        min-height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 12px 30px rgba(0,0,0,0.2),
            0 8px 20px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, 
            rgba(26, 35, 126, 0.95) 0%, 
            rgba(48, 63, 159, 0.95) 100%) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        color: white;
        border-right: 2px solid rgba(255,255,255,0.12);
        padding: 1.5rem 1rem;
    }
    
    /* Upload Section */
    .upload-section-container {
        border: 3px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
        min-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .upload-section-container:hover {
        border-color: rgba(118, 75, 162, 0.7);
        background: rgba(118, 75, 162, 0.08);
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    }
    
    .upload-icon {
        font-size: 3.5rem;
        color: rgba(102, 126, 234, 0.7);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .upload-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.8rem;
    }
    
    .upload-subtitle {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
        line-height: 1.5;
        max-width: 400px;
    }
    
    .upload-hint {
        font-size: 0.85rem;
        color: #888;
        margin-top: 1.5rem;
        font-style: italic;
    }
    
    /* Camera Section */
    .camera-section-container {
        border: 3px dashed rgba(38, 166, 154, 0.5);
        border-radius: 20px;
        padding: 2.5rem 1.8rem;
        text-align: center;
        background: rgba(38, 166, 154, 0.05);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
        min-height: 320px;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .camera-section-container:hover {
        border-color: rgba(38, 166, 154, 0.7);
        background: rgba(38, 166, 154, 0.1);
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    }
    
    .camera-icon {
        font-size: 3.5rem;
        color: rgba(38, 166, 154, 0.7);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    
    /* Image Preview */
    .image-preview-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border: 2px solid rgba(255,255,255,0.8);
        background: white;
    }
    
    .image-preview-container img {
        width: 100%;
        height: auto;
        display: block;
        transition: transform 0.3s ease;
    }
    
    .image-preview-container img:hover {
        transform: scale(1.02);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 0.9rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        color: #555;
        border: 1.5px solid rgba(0,0,0,0.08);
        transition: all 0.25s ease;
        margin: 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        background: white;
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: rgba(255,255,255,0.3) !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Progress Bar */
    .confidence-bar {
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 0.7rem;
        margin-top: 1.2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .confidence-fill {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        transition: width 1.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem;
        background: rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 22px;
        margin-top: 3rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes moveBackground {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    @keyframes rainbow {
        0% {
            background-position: 0% 50%;
        }
        100% {
            background-position: 200% 50%;
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8));
    }
    
    /* Selection */
    ::selection {
        background: rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
        width: 30px !important;
        height: 30px !important;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3.5rem 2rem;
        color: #666;
    }
    
    .empty-state h3 {
        font-size: 1.7rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #444;
    }
    
    .empty-state p {
        font-size: 1.05rem;
        opacity: 0.7;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Disease Severity Badge */
    .severity-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-left: 0.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .severity-high {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
    }
    
    .severity-medium {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #333;
    }
    
    .severity-low {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }
    
    /* History Card */
    .history-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
        transition: all 0.25s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .history-card:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border-left-color: #764ba2;
    }
    
    /* Form Elements */
    .stFileUploader {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: rgba(102, 126, 234, 0.03) !important;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2 !important;
        background: rgba(118, 75, 162, 0.05) !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.4rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .card {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1.4rem 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .prediction-disease {
            font-size: 2.2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.7rem 1.2rem;
            font-size: 0.9rem;
        }
        
        .upload-section-container,
        .camera-section-container {
            padding: 2rem 1.5rem;
            min-height: 280px;
        }
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #667eea;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(0, 0, 0, 0.85);
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL CONFIGURATION ====================
CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut']
MODEL_PATH = 'models/best_model.pth'
IMG_SIZE = 224

DISEASE_INFO = {
    'Bacterialblight': {
        'name': 'Bacterial Blight',
        'scientific': 'Xanthomonas oryzae pv. oryzae',
        'symptoms': [
            'Water-soaked lesions on leaf margins',
            'Yellow to white lesions with wavy edges',
            'Systemic infection leads to wilting',
            'Bacterial ooze visible in morning'
        ],
        'treatment': [
            'Use resistant varieties',
            'Apply copper-based bactericides',
            'Maintain field sanitation',
            'Avoid excessive nitrogen fertilizer'
        ],
        'severity': 'HIGH',
        'color': '#ef5350'
    },
    'Brownspot': {
        'name': 'Brown Spot',
        'scientific': 'Bipolaris oryzae',
        'symptoms': [
            'Circular brown spots with gray centers',
            'Spots surrounded by yellow halos',
            'Numerous spots cause leaf blight',
            'Affects yield and grain quality'
        ],
        'treatment': [
            'Apply fungicides (Mancozeb, Tricyclazole)',
            'Balance nutrient management',
            'Ensure proper drainage',
            'Use disease-free seeds'
        ],
        'severity': 'MEDIUM',
        'color': '#ff9800'
    },
    'Leafsmut': {
        'name': 'Leaf Smut',
        'scientific': 'Entyloma oryzae',
        'symptoms': [
            'Small black angular spots on leaves',
            'Spots aligned between leaf veins',
            'Severe infection causes leaf yellowing',
            'Reduces photosynthesis efficiency'
        ],
        'treatment': [
            'Apply systemic fungicides early',
            'Remove infected plant debris',
            'Crop rotation with non-host crops',
            'Improve air circulation in field'
        ],
        'severity': 'LOW-MEDIUM',
        'color': '#66bb6a'
    }
}

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load the trained ResNet-34 model"""
    class ResNet34RiceClassifier(nn.Module):
        def __init__(self, num_classes=3):
            super(ResNet34RiceClassifier, self).__init__()
            self.model = models.resnet34(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)
    
    model = ResNet34RiceClassifier(num_classes=len(CLASSES))
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

@st.cache_resource
def load_metrics():
    """Load training metrics"""
    metrics_path = 'results/metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading metrics: {e}")
            return None
    return None

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# ==================== CAMERA FUNCTIONS ====================
def capture_from_camera():
    """Capture image from webcam"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your camera permissions and connections.")
            return None
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            st.error("Failed to capture image from camera.")
            return None
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def process_camera_image(image):
    """Process camera image for display"""
    if image is None:
        return None
    
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if img_array is None or img_array.size == 0:
            return image  # Return original if empty
        
        # Add some image enhancement
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(np.clip(enhanced, 0, 255).astype('uint8'))
    except Exception as e:
        # Return original image if processing fails
        return image

# ==================== PREDICTION ====================
def predict_disease(model, image):
    """Predict disease from image"""
    try:
        with torch.no_grad():
            image_tensor = preprocess_image(image)
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            results = {
                'class': CLASSES[predicted_class],
                'confidence': confidence,
                'all_probabilities': {
                    CLASSES[i]: float(probabilities[i]) 
                    for i in range(len(CLASSES))
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return results
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ==================== VISUALIZATION FUNCTIONS ====================
def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Confidence Level",
            'font': {'size': 20, 'color': 'white', 'family': 'Poppins'}
        },
        delta={'reference': 80, 'increasing': {'color': "#38ef7d"}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1.5,
                'tickcolor': "white",
                'tickfont': {'color': 'white', 'size': 11}
            },
            'bar': {'color': "#1a237e"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 1.5,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 83, 80, 0.5)'},
                {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.5)'},
                {'range': [75, 100], 'color': 'rgba(102, 187, 106, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Poppins"},
        height=280,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_probability_chart(probabilities):
    """Create probability bar chart"""
    df = pd.DataFrame(list(probabilities.items()), 
                     columns=['Disease', 'Probability'])
    df['Probability'] = df['Probability'] * 100
    df['Disease_Name'] = df['Disease'].map(lambda x: DISEASE_INFO[x]['name'])
    
    fig = px.bar(
        df, x='Probability', y='Disease_Name',
        orientation='h',
        color='Probability',
        color_continuous_scale='Viridis',
        labels={'Probability': 'Probability (%)', 'Disease_Name': 'Disease'},
        title='Prediction Probabilities',
        text='Probability'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.95)',
        font={'family': 'Poppins', 'size': 12},
        title_font={'size': 18, 'color': '#2c3e50', 'family': 'Poppins'},
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    fig.update_traces(
        marker_line_color='rgb(30, 30, 30)',
        marker_line_width=1.2,
        texttemplate='%{x:.1f}%',
        textposition='outside',
        textfont={'size': 11, 'color': '#2c3e50'}
    )
    
    fig.update_xaxes(range=[0, 100])
    
    return fig

def create_disease_severity_chart():
    """Create disease severity comparison chart"""
    disease_names = [DISEASE_INFO[disease]['name'] for disease in CLASSES]
    severities = []
    colors = []
    descriptions = []
    
    for disease in CLASSES:
        info = DISEASE_INFO[disease]
        severity_value = {
            'HIGH': 90,
            'MEDIUM': 65,
            'LOW-MEDIUM': 40
        }.get(info['severity'], 50)
        severities.append(severity_value)
        colors.append(info['color'])
        descriptions.append(info['severity'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=disease_names,
            y=severities,
            marker_color=colors,
            text=descriptions,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Severity: %{text}<br>Level: %{y}%<extra></extra>',
            marker_line_color='rgba(0,0,0,0.3)',
            marker_line_width=1.5
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Disease Severity Comparison',
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Poppins'}
        },
        xaxis_title={
            'text': 'Disease',
            'font': {'size': 13, 'color': '#555'}
        },
        yaxis_title={
            'text': 'Severity Level (%)',
            'font': {'size': 13, 'color': '#555'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.95)',
        font={'family': 'Poppins', 'size': 12},
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    fig.update_yaxes(range=[0, 100])
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🌾AgroSight</h1>
        <p class="hero-subtitle">Advanced AI-Powered Rice Disease Detection System</p>
        <p class="hero-tagline">
            Empower your agricultural decisions with cutting-edge deep learning technology. 
            Detect, analyze, and manage rice leaf diseases with precision and confidence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    metrics = load_metrics()
    
    if model is None:
        st.error("""
        ## Model Not Found
        The trained model file was not found. Please ensure:
        1. You have trained the model using `train_model.py`
        2. The model file exists at: `models/best_model.pth`
        3. You have the required dependencies installed
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 0.5rem;">
                Navigation
            </div>
            <div style="height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); 
                 border-radius: 2px; margin: 0 auto; width: 60px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Select Page",
            ["Disease Detection", "Model Performance", "Disease Information", 
             "Analysis History", "About"],
            label_visibility="collapsed"
        )
        
        # Extract page name from selection
        page_name = page.split(" ")[-1]
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Status indicators
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
                <div class="metric-label">Model Status</div>
                <div class="metric-value" style="font-size: 1.4rem; color: #38ef7d;">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem; margin-bottom: 1rem;">
                <div class="metric-label">Diseases</div>
                <div class="metric-value" style="font-size: 1.4rem;">{len(CLASSES)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if metrics:
            st.markdown("### Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                accuracy = metrics.get('test_accuracy', 0) * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")
            with col2:
                if 'classification_report' in metrics:
                    avg_f1 = np.mean([metrics['classification_report'][cls]['f1-score'] for cls in CLASSES]) * 100
                else:
                    avg_f1 = 0
                st.metric("Avg F1-Score", f"{avg_f1:.1f}%")
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("Clear History", key="clear_history", use_container_width=True):
            if 'history' in st.session_state:
                st.session_state.history = []
                st.success("History cleared successfully!")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.08); 
             border-radius: 12px; margin-top: 1rem;'>
            <p style='color: white; font-size: 0.9rem; margin-bottom: 0.3rem; font-weight: 500;'>
                System Version 2.1.0
            </p>
            <p style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>
                Powered by ResNet-34 AI
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Routing
    if page_name == "Detection":
        show_detection_page(model)
    elif page_name == "Performance":
        show_performance_page(metrics)
    elif page_name == "Information":
        show_disease_info_page()
    elif page_name == "History":
        show_history_page()
    else:
        show_about_page()

def show_detection_page(model):
    """Disease detection page"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Disease Detection</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; margin-bottom: 1.5rem; font-size: 1.05rem;">Upload or capture an image of a rice leaf for AI-powered disease analysis.</p>', unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Capture from Camera"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Custom upload section
            st.markdown("""
            <div class="upload-section-container">
                <div class="upload-icon">📤</div>
                <div class="upload-title">Upload Rice Leaf Image</div>
                <div class="upload-subtitle">
                    Drag and drop your image here or click to browse.<br>
                    Supported formats: JPG, JPEG, PNG
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Select a clear image of a rice leaf for accurate analysis",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                    st.image(image, caption='Uploaded Image Preview', use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("Analyze Uploaded Image", key="analyze_upload", use_container_width=True):
                        with st.spinner("AI is analyzing the image..."):
                            results = predict_disease(model, image)
                            
                            if results:
                                # Add to history
                                if 'history' not in st.session_state:
                                    st.session_state.history = []
                                st.session_state.history.append({
                                    **results,
                                    'source': 'upload',
                                    'filename': uploaded_file.name if hasattr(uploaded_file, 'name') else 'Uploaded Image',
                                    'image': image
                                })
                                
                                st.session_state['results'] = results
                                st.session_state['analyzed'] = True
                                st.session_state['image_source'] = 'upload'
                                st.rerun()
                            else:
                                st.error("Failed to analyze image. Please try again.")
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        with col2:
            if 'analyzed' in st.session_state and st.session_state.get('analyzed') and st.session_state.get('image_source') == 'upload':
                results = st.session_state.get('results', {})
                if results:
                    disease = results.get('class', 'Unknown')
                    confidence = results.get('confidence', 0)
                    disease_name = DISEASE_INFO.get(disease, {}).get('name', disease)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-title">Detection Result</div>
                        <div class="prediction-disease">{disease_name}</div>
                        <div style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.95;">
                            Confidence Level: <strong>{confidence*100:.2f}%</strong>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.95rem; opacity: 0.85;">
                            Detection Time: {results.get('timestamp', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key="detection_gauge"  # Fixed: Added unique key
                    )
    
    with tab2:
        st.markdown("""
        <div class="camera-section-container">
            <div class="camera-icon">📸</div>
            <div class="upload-title">Real-time Camera Capture</div>
            <div class="upload-subtitle">
                Capture a photo of rice leaf using your webcam.<br>
                Ensure good lighting and clear focus for best results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Capture Image", key="capture", use_container_width=True):
                with st.spinner("Accessing camera..."):
                    captured_image = capture_from_camera()
                    if captured_image:
                        st.session_state['captured_image'] = captured_image
                        st.session_state['captured_image_display'] = captured_image
                        st.success("Image captured successfully!")
                    else:
                        st.error("Failed to capture image. Please check your camera.")
        
        with col2:
            if 'captured_image_display' in st.session_state and st.session_state['captured_image_display']:
                try:
                    processed_image = process_camera_image(st.session_state['captured_image_display'])
                    if processed_image:
                        st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                        st.image(processed_image, caption='Captured Image Preview', use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if st.button("Analyze Captured Image", key="analyze_capture", use_container_width=True):
                            if 'captured_image' in st.session_state:
                                with st.spinner("AI is analyzing the captured image..."):
                                    results = predict_disease(model, st.session_state['captured_image'])
                                    
                                    if results:
                                        # Add to history
                                        if 'history' not in st.session_state:
                                            st.session_state.history = []
                                        st.session_state.history.append({
                                            **results,
                                            'source': 'camera',
                                            'filename': 'Camera Capture',
                                            'image': st.session_state['captured_image']
                                        })
                                        
                                        st.session_state['results'] = results
                                        st.session_state['analyzed'] = True
                                        st.session_state['image_source'] = 'camera'
                                        st.rerun()
                                    else:
                                        st.error("Failed to analyze captured image. Please try again.")
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show detailed results
    if 'analyzed' in st.session_state and st.session_state.get('analyzed'):
        results = st.session_state.get('results', {})
        if results:
            disease = results.get('class', '')
            disease_info = DISEASE_INFO.get(disease, {})
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">Detailed Analysis Report</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if 'all_probabilities' in results:
                    st.plotly_chart(
                        create_probability_chart(results['all_probabilities']),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key="probability_chart"  # Fixed: Added unique key
                    )
                
                if disease_info:
                    severity = disease_info.get('severity', 'UNKNOWN')
                    severity_class = {
                        'HIGH': 'severity-high',
                        'MEDIUM': 'severity-medium',
                        'LOW-MEDIUM': 'severity-low'
                    }.get(severity, 'severity-medium')
                    
                    st.markdown("### Disease Severity Assessment")
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 1rem 0; padding: 1.2rem; 
                         background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,245,245,0.9)); 
                         border-radius: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                        <span style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-right: 1rem;">
                            Severity Level:
                        </span>
                        <span class="severity-badge {severity_class}">
                            {severity}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if disease_info:
                    st.markdown(f"""
                    <div class="disease-info">
                        <h3 style="margin-bottom: 1rem;">{disease_info.get('name', 'Unknown Disease')}</h3>
                        <p><strong>Scientific Name:</strong> {disease_info.get('scientific', 'Unknown')}</p>
                        <p><strong>Confidence Score:</strong> {results.get('confidence', 0)*100:.2f}%</p>
                        <p><strong>Detection Time:</strong> {results.get('timestamp', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Key Symptoms")
                    for symptom in disease_info.get('symptoms', []):
                        st.markdown(f"""
                        <div class="history-card" style="padding: 1rem; margin-bottom: 0.7rem;">
                            <div style="display: flex; align-items: start;">
                                <div style="margin-right: 0.7rem; color: #ff4444; font-size: 1.2rem;"></div>
                                <div style="font-size: 0.98rem; color: #444;">{symptom}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### Recommended Treatment")
                    for treatment in disease_info.get('treatment', []):
                        st.markdown(f"""
                        <div class="history-card" style="padding: 1rem; margin-bottom: 0.7rem; border-left-color: #4CAF50;">
                            <div style="display: flex; align-items: start;">
                                <div style="margin-right: 0.7rem; color: #4CAF50; font-size: 1.2rem;"></div>
                                <div style="font-size: 0.98rem; color: #444;">{treatment}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_performance_page(metrics):
    """Model performance page"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if metrics is None:
        st.info("""
        ### No Performance Data Available
        Model performance metrics are not currently available. 
        Please train the model first to see detailed performance analytics.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Overall Metrics
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    accuracy = metrics.get('test_accuracy', 0) * 100
    if 'classification_report' in metrics:
        avg_precision = np.mean([metrics['classification_report'][cls]['precision'] for cls in CLASSES]) * 100
        avg_recall = np.mean([metrics['classification_report'][cls]['recall'] for cls in CLASSES]) * 100
        avg_f1 = np.mean([metrics['classification_report'][cls]['f1-score'] for cls in CLASSES]) * 100
    else:
        avg_precision = avg_recall = avg_f1 = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{accuracy:.1f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                Overall prediction accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{avg_precision:.1f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                Correct positive predictions
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{avg_recall:.1f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                True positives identified
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{avg_f1:.1f}%</div>
            <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                Balance of precision & recall
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Classification Report
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Per-Class Performance Analysis</h2>', unsafe_allow_html=True)
    
    if 'classification_report' in metrics:
        class_metrics = []
        for cls in CLASSES:
            report = metrics['classification_report'][cls]
            class_metrics.append({
                'Disease': DISEASE_INFO[cls]['name'],
                'Precision': f"{report['precision']*100:.2f}%",
                'Recall': f"{report['recall']*100:.2f}%",
                'F1-Score': f"{report['f1-score']*100:.2f}%",
                'Support': report['support']
            })
        
        df = pd.DataFrame(class_metrics)
        
        # Apply custom styling
        def color_cells(val):
            if isinstance(val, str) and '%' in val:
                num = float(val.replace('%', ''))
                if num >= 90:
                    return 'background-color: rgba(102, 187, 106, 0.2); color: #2e7d32; font-weight: 600;'
                elif num >= 80:
                    return 'background-color: rgba(255, 193, 7, 0.2); color: #f57f17; font-weight: 500;'
                else:
                    return 'background-color: rgba(239, 83, 80, 0.2); color: #c62828; font-weight: 500;'
            return ''
        
        styled_df = df.style.applymap(color_cells, subset=['Precision', 'Recall', 'F1-Score'])
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=250,
            column_config={
                "Disease": st.column_config.TextColumn("Disease", width="medium"),
                "Precision": st.column_config.TextColumn("Precision", width="small"),
                "Recall": st.column_config.TextColumn("Recall", width="small"),
                "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
                "Support": st.column_config.NumberColumn("Samples", width="small")
            }
        )
    else:
        st.warning("Classification report data not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Performance Visualizations</h2>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    
    with cols[0]:
        if 'classification_report' in metrics:
            fig1 = go.Figure()
            
            diseases = [DISEASE_INFO[cls]['name'] for cls in CLASSES]
            precision_vals = [metrics['classification_report'][cls]['precision']*100 for cls in CLASSES]
            recall_vals = [metrics['classification_report'][cls]['recall']*100 for cls in CLASSES]
            
            fig1.add_trace(go.Bar(
                name='Precision',
                x=diseases,
                y=precision_vals,
                marker_color='#667eea',
                text=[f"{v:.1f}%" for v in precision_vals],
                textposition='outside'
            ))
            
            fig1.add_trace(go.Bar(
                name='Recall',
                x=diseases,
                y=recall_vals,
                marker_color='#764ba2',
                text=[f"{v:.1f}%" for v in recall_vals],
                textposition='outside'
            ))
            
            fig1.update_layout(
                title={
                    'text': 'Precision & Recall by Disease',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                barmode='group',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.95)',
                font={'family': 'Poppins', 'size': 12},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            fig1.update_yaxes(range=[0, 100], title_text="Score (%)")
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False}, key="precision_recall_chart")  # Fixed: Added unique key
    
    with cols[1]:
        st.plotly_chart(
            create_disease_severity_chart(),
            use_container_width=True,
            config={'displayModeBar': False},
            key="severity_chart"  # Fixed: Added unique key
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_disease_info_page():
    """Disease information page"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Rice Disease Information Library</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; margin-bottom: 2rem; font-size: 1.05rem;">Comprehensive information about common rice diseases, their symptoms, and treatment methods.</p>', unsafe_allow_html=True)
    
    for disease, info in DISEASE_INFO.items():
        with st.expander(f"{info['name']} - {info['scientific']}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Disease Characteristics")
                
                severity_class = {
                    'HIGH': 'severity-high',
                    'MEDIUM': 'severity-medium',
                    'LOW-MEDIUM': 'severity-low'
                }.get(info['severity'], 'severity-medium')
                
                st.markdown(f"""
                <div style="margin-bottom: 1.5rem; padding: 1.2rem; background: linear-gradient(135deg, 
                    rgba(255,255,255,0.9), rgba(245,245,245,0.9)); border-radius: 14px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-weight: 600; margin-right: 0.8rem; color: #2c3e50;">Severity:</span>
                        <span class="severity-badge {severity_class}">{info['severity']}</span>
                    </div>
                    <div style="color: #555; font-size: 0.95rem; line-height: 1.5;">
                        This disease is classified as <strong>{info['severity']}</strong> severity, 
                        requiring appropriate management strategies.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Common Symptoms")
                for symptom in info['symptoms']:
                    st.markdown(f"""
                    <div class="history-card" style="margin-bottom: 0.7rem; padding: 1rem;">
                        <div style="display: flex; align-items: start;">
                            <div style="margin-right: 0.7rem; color: #ff4444; font-size: 1.1rem;"></div>
                            <div style="font-size: 0.98rem; color: #444;">{symptom}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Management & Control")
                
                st.markdown("#### Treatment Recommendations")
                for treatment in info['treatment']:
                    st.markdown(f"""
                    <div class="history-card" style="margin-bottom: 0.7rem; padding: 1rem; border-left-color: #4CAF50;">
                        <div style="display: flex; align-items: start;">
                            <div style="margin-right: 0.7rem; color: #4CAF50; font-size: 1.1rem;"></div>
                            <div style="font-size: 0.98rem; color: #444;">{treatment}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Preventive Measures")
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(66, 165, 245, 0.05)); 
                     padding: 1.2rem; border-radius: 12px; margin-top: 1rem; border: 1px solid rgba(33, 150, 243, 0.2);">
                    <p style="margin-bottom: 0.8rem; font-weight: 600; color: #2196F3; font-size: 1.05rem;">
                        General Prevention Tips:
                    </p>
                    <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
                        • Regular field monitoring and inspection
                    </p>
                    <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
                        • Proper irrigation and water management
                    </p>
                    <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
                        • Use of certified disease-free seeds
                    </p>
                    <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
                        • Crop rotation and field sanitation practices
                    </p>
                    <p style="font-size: 0.95rem; color: #555; margin-bottom: 0.4rem;">
                        • Balanced fertilizer application
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_history_page():
    """Analysis history page"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Analysis History</h2>', unsafe_allow_html=True)
    
    if 'history' not in st.session_state or len(st.session_state.history) == 0:
        st.markdown("""
        <div class="empty-state">
            <h3>No Analysis History Found</h3>
            <p>You haven't analyzed any images yet. Start by uploading or capturing a rice leaf image to see your history here.</p>
            <div style="margin-top: 2rem;">
                <a href="#disease-detection">
                    <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; border: none; padding: 0.9rem 2.2rem; 
                            border-radius: 12px; cursor: pointer; font-weight: 600;
                            font-size: 1rem; transition: all 0.3s ease;">
                        Go to Detection Page
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Sort history by timestamp (newest first)
        history = sorted(st.session_state.history, 
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True)
        
        st.markdown(f"<p style='color: #666; margin-bottom: 1.5rem;'>Total analyses: <strong>{len(history)}</strong></p>", unsafe_allow_html=True)
        
        for idx, record in enumerate(history):
            disease = record.get('class', '')
            disease_info = DISEASE_INFO.get(disease, {})
            disease_name = disease_info.get('name', 'Unknown Disease')
            severity = disease_info.get('severity', 'UNKNOWN')
            
            severity_class = {
                'HIGH': 'severity-high',
                'MEDIUM': 'severity-medium',
                'LOW-MEDIUM': 'severity-low'
            }.get(severity, 'severity-medium')
            
            with st.expander(f"Analysis #{idx+1}: {disease_name} - {record.get('timestamp', 'Unknown time')}", 
                           expanded=idx < 2):  # Expand first 2 by default
                col1, col2 = st.columns([1, 2])
                
            with col1:
                confidence = record.get('confidence', 0) * 100
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        padding: 1.5rem;
                        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(245,245,245,0.95));
                        border-radius: 16px;
                        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
                    ">
                        <div style="
                            font-size: 2.5rem;
                            font-weight: 800;
                            background: linear-gradient(45deg, #667eea, #764ba2);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                        ">
                            {confidence:.1f}%
                        </div>
                        <div style="
                            margin-top: 0.4rem;
                            font-size: 0.9rem;
                            color: #666;
                        ">
                            Confidence Score
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with col2:
                    st.markdown(f"**Disease:** {disease_name}")
                    if disease_info.get('scientific'):
                        st.markdown(f"**Scientific Name:** {disease_info['scientific']}")
                    st.markdown(f"**Detection Time:** {record.get('timestamp', 'Unknown')}")
                    
                    if 'all_probabilities' in record:
                        st.markdown("**Probability Distribution:**")
                        prob_df = pd.DataFrame(list(record['all_probabilities'].items()),
                                             columns=['Disease', 'Probability'])
                        prob_df['Probability'] = prob_df['Probability'] * 100
                        prob_df['Disease_Name'] = prob_df['Disease'].map(lambda x: DISEASE_INFO.get(x, {}).get('name', x))
                        
                        # Create a mini bar chart
                        fig = px.bar(
                            prob_df.sort_values('Probability', ascending=True),
                            x='Probability',
                            y='Disease_Name',
                            orientation='h',
                            color='Probability',
                            color_continuous_scale='Viridis',
                            range_x=[0, 100],
                            height=200
                        )
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(255,255,255,0.9)',
                            font={'family': 'Poppins', 'size': 10},
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            xaxis_showgrid=False,
                            yaxis_showgrid=False
                        )
                        
                        fig.update_traces(
                            marker_line_width=0.5,
                            texttemplate='%{x:.1f}%',
                            textposition='outside',
                            textfont={'size': 9}
                        )
                        
                        # Fixed: Added unique key for each chart in the loop
                        st.plotly_chart(fig, 
                            use_container_width=True, 
                            config={'displayModeBar': False},
                            key=f"history_prob_{idx}"  # This fixes the error
                        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_about_page():
    """About page"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">About AgroSight</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Overview
        **AgroSight** is an advanced AI-powered platform designed to revolutionize rice disease detection 
        using state-of-the-art deep learning technology. Our system employs a sophisticated ResNet-34 
        convolutional neural network architecture to accurately classify three major rice diseases with 
        exceptional precision and reliability.
        
        ### Core Features
        - **Real-time AI Detection**: Instant disease predictions from uploaded or captured images
        - **High Accuracy**: Trained on extensive datasets of rice leaf images for reliable results
        - **Comprehensive Analysis**: Detailed disease information and evidence-based treatment recommendations
        - **User-Friendly Interface**: Accessible to farmers, agronomists, researchers, and enthusiasts
        - **Historical Tracking**: Maintains detailed analysis history for monitoring and comparison
        
        ### Technology Architecture
        """)
        
        tech_stack = {
            "Deep Learning Framework": "PyTorch",
            "Neural Network": "ResNet-34 Architecture",
            "Web Framework": "Streamlit",
            "Visualization": "Plotly & Matplotlib",
            "Image Processing": "OpenCV & PIL",
            "Data Handling": "Pandas & NumPy"
        }
        
        for tech, desc in tech_stack.items():
            st.markdown(f"""
            <div class="history-card" style="margin-bottom: 0.6rem;">
                <div style="display: flex; align-items: center;">
                    <div style="margin-right: 0.7rem; font-size: 1.2rem;"></div>
                    <div>
                        <strong>{tech}</strong>: {desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Detected Diseases
        
        **1. Bacterial Blight**
        - **Pathogen**: *Xanthomonas oryzae* pv. *oryzae*
        - **Impact**: Can cause up to 70% yield loss in severe cases
        - **Season**: Prevalent in wet seasons with high humidity
        
        **2. Brown Spot**
        - **Pathogen**: *Bipolaris oryzae*
        - **Impact**: Reduces grain quality and overall yield significantly
        - **Season**: Common in nutrient-deficient soils and stressful conditions
        
        **3. Leaf Smut**
        - **Pathogen**: *Entyloma oryzae*
        - **Impact**: Affects photosynthesis efficiency and plant vitality
        - **Season**: Found in humid conditions with poor air circulation
        
        ### Mission & Vision
        
        **Our Mission:**
        To empower farmers and agricultural professionals with accessible, accurate, 
        and timely disease detection tools, supporting sustainable agriculture practices 
        and strengthening global food security through technological innovation.
        
        **Our Vision:**
        To become the world's leading AI-powered agricultural diagnostic platform, 
        revolutionizing crop disease management and contributing to food sustainability worldwide.
        
        ### Development Team
        This innovative platform is developed by a multidisciplinary team of AI researchers, 
        agricultural scientists, and software engineers dedicated to bridging the gap between 
        advanced technology and practical agricultural needs.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### Contact & Support
        
        For technical support, feature requests, or collaboration opportunities:
        
        - **Email**: support@agrosight.ai
        - **Documentation**: docs.agrosight.ai
        - **GitHub**: github.com/agrosight
        - **Research Papers**: research.agrosight.ai
        
        ### Version Information
        - **Current Version**: 2.1.0
        - **Last Updated**: 2025
        - **Model Version**: ResNet-34 v1.2
        - **Data Version**: RiceLeaf-2024
        - **Release Status**: Production Ready
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.8rem; 
           background: linear-gradient(45deg, #ffffff, #e8f5e9); 
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
           background-clip: text;">
           AgroSight - Revolutionizing Agricultural Diagnostics
        </p>
        <p style="font-size: 1rem; margin-bottom: 0.8rem; opacity: 0.9; color: rgba(255,255,255,0.9);">
           Advanced AI for Sustainable Agriculture & Food Security
        </p>
        <div style="margin-top: 1.8rem; display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">>95%</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">Accuracy</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;"><2s</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">Response Time</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">3+</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">Diseases</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #38ef7d;">24/7</div>
                <div style="font-size: 0.85rem; opacity: 0.8;">Availability</div>
            </div>
        </div>
        <p style="font-size: 0.85rem; margin-top: 2rem; opacity: 0.7; color: rgba(255,255,255,0.7);">
            © 2025 AgroSight Research Lab | Powered by PyTorch & Streamlit | All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'captured_image' not in st.session_state:
        st.session_state['captured_image'] = None
    if 'captured_image_display' not in st.session_state:
        st.session_state['captured_image_display'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = {}
    if 'image_source' not in st.session_state:
        st.session_state['image_source'] = None
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext")
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    
    main()
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time

# Configure page
st.set_page_config(
    page_title="🛡️ Cyber Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .threat-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Model class definition
class CyberThreatModel(nn.Module):
    def __init__(self, input_size):
        super(CyberThreatModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

feature_columns = ['processId', 'threadId', 'parentProcessId', 'userId', 
                  'mountNamespace', 'argsNum', 'returnValue']

def initialize_model():
    """Initialize model and scaler"""
    st.session_state.model = CyberThreatModel(len(feature_columns))
    st.session_state.scaler = StandardScaler()

def train_model(train_df, val_df, epochs=10, learning_rate=1e-3):
    """Train the model with progress tracking"""
    try:
        # Prepare data
        X_train = train_df[feature_columns].values
        y_train = train_df['sus_label'].values
        X_val = val_df[feature_columns].values
        y_val = val_df['sus_label'].values
        
        # Scale data
        X_train_scaled = st.session_state.scaler.fit_transform(X_train)
        X_val_scaled = st.session_state.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        # Initialize model
        st.session_state.model = CyberThreatModel(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(st.session_state.model.parameters(), 
                                   lr=learning_rate, weight_decay=1e-4)
        
        # Training with progress tracking
        training_losses = []
        validation_losses = []
        validation_accuracies = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            # Training
            st.session_state.model.train()
            optimizer.zero_grad()
            train_outputs = st.session_state.model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            st.session_state.model.eval()
            with torch.no_grad():
                val_outputs = st.session_state.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == y_val_tensor).float().mean()
            
            # Store metrics
            training_losses.append(float(train_loss))
            validation_losses.append(float(val_loss))
            validation_accuracies.append(float(val_accuracy))
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch + 1}/{epochs} - '
                           f'Train Loss: {train_loss:.4f}, '
                           f'Val Loss: {val_loss:.4f}, '
                           f'Val Acc: {val_accuracy:.4f}')
        
        progress_bar.progress(1.0)
        status_text.text("Training completed successfully!")
        
        # Store training history
        st.session_state.training_history = {
            'epochs': list(range(1, epochs + 1)),
            'train_loss': training_losses,
            'val_loss': validation_losses,
            'val_accuracy': validation_accuracies
        }
        
        st.session_state.model_trained = True
        return True, float(val_accuracy)
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False, 0.0

def predict_threats(data_df):
    """Make predictions on data"""
    if st.session_state.model is None or st.session_state.scaler is None:
        return None, "Model not trained yet!"
    
    try:
        # Prepare data
        X = data_df[feature_columns].values
        X_scaled = st.session_state.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Make predictions
        st.session_state.model.eval()
        with torch.no_grad():
            probabilities = st.session_state.model(X_tensor).numpy()
            predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions, probabilities
        
    except Exception as e:
        return None, f"Prediction failed: {str(e)}"

# Main app
def main():
    st.markdown('<h1 class="main-header">🛡️ Cyber Threat Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>
        Advanced deep learning model for detecting cyber threats in system logs
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Model Control")
        
        # Model status
        if st.session_state.model_trained:
            st.success("✅ Model Trained & Ready")
        else:
            st.warning("⚠️ Model Not Trained")
        
        st.markdown("---")
        
        # Data format info
        st.header("📋 Data Format")
        st.markdown("""
        **Required Columns:**
        - processId
        - threadId  
        - parentProcessId
        - userId
        - mountNamespace
        - argsNum
        - returnValue
        - sus_label (training only)
        """)
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.training_history:
            st.header("📊 Model Stats")
            final_accuracy = st.session_state.training_history['val_accuracy'][-1]
            st.metric("Validation Accuracy", f"{final_accuracy:.2%}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏋️ Train Model", "🔍 Detect Threats", "📈 Analytics", "🧪 Single Prediction"])
    
    with tab1:
        st.header("🏋️ Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Training Data")
            train_file = st.file_uploader("Choose training CSV file", 
                                        type=['csv'], key="train")
            
        with col2:
            st.subheader("📤 Upload Validation Data")  
            val_file = st.file_uploader("Choose validation CSV file", 
                                      type=['csv'], key="val")
        
        # Training parameters
        st.subheader("⚙️ Training Parameters")
        col3, col4 = st.columns(2)
        with col3:
            epochs = st.slider("Epochs", 5, 50, 10)
        with col4:
            learning_rate = st.select_slider("Learning Rate", 
                                           options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                           value=1e-3,
                                           format_func=lambda x: f"{x:.0e}")
        
        if st.button("🚀 Train Model", type="primary"):
            if train_file and val_file:
                with st.spinner("Loading data..."):
                    try:
                        train_df = pd.read_csv(train_file)
                        val_df = pd.read_csv(val_file)
                        
                        # Validate columns
                        required_cols = feature_columns + ['sus_label']
                        train_missing = [col for col in required_cols if col not in train_df.columns]
                        val_missing = [col for col in required_cols if col not in val_df.columns]
                        
                        if train_missing or val_missing:
                            st.error(f"Missing columns - Train: {train_missing}, Val: {val_missing}")
                        else:
                            st.success("✅ Data loaded successfully!")
                            
                            # Display data info
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Train Records", len(train_df))
                            with col2:
                                st.metric("Val Records", len(val_df))
                            with col3:
                                train_threats = train_df['sus_label'].sum()
                                st.metric("Train Threats", train_threats)
                            with col4:
                                val_threats = val_df['sus_label'].sum()
                                st.metric("Val Threats", val_threats)
                            
                            # Train model
                            st.subheader("🏃 Training Progress")
                            success, accuracy = train_model(train_df, val_df, epochs, learning_rate)
                            
                            if success:
                                st.balloons()
                                st.success(f"🎉 Training completed! Final validation accuracy: {accuracy:.2%}")
                    
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("Please upload both training and validation files.")
    
    with tab2:
        st.header("🔍 Threat Detection")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Train Model' tab.")
        else:
            predict_file = st.file_uploader("Choose CSV file for threat detection", 
                                          type=['csv'], key="predict")
            
            if predict_file:
                try:
                    predict_df = pd.read_csv(predict_file)
                    
                    # Validate columns
                    missing_cols = [col for col in feature_columns if col not in predict_df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                    else:
                        st.success("✅ File loaded successfully!")
                        
                        with st.spinner("🔍 Analyzing threats..."):
                            predictions, probabilities = predict_threats(predict_df)
                            
                            if predictions is not None:
                                # Calculate statistics
                                total_records = len(predictions)
                                threats_detected = int(np.sum(predictions))
                                safe_records = total_records - threats_detected
                                threat_percentage = (threats_detected / total_records) * 100
                                
                                # Display results
                                st.subheader("📊 Analysis Results")
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Records", total_records)
                                with col2:
                                    st.metric("🚨 Threats", threats_detected, 
                                            delta=f"{threat_percentage:.1f}%")
                                with col3:
                                    st.metric("✅ Safe", safe_records)
                                with col4:
                                    st.metric("Risk Level", 
                                            "HIGH" if threat_percentage > 10 else 
                                            "MEDIUM" if threat_percentage > 5 else "LOW")
                                
                                # Alert box
                                if threats_detected > 0:
                                    st.markdown(f"""
                                    <div class="threat-alert">
                                        🚨 <strong>THREATS DETECTED!</strong><br>
                                        {threats_detected} suspicious activities found ({threat_percentage:.1f}% of total)
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="safe-alert">
                                        ✅ <strong>ALL CLEAR!</strong><br>
                                        No threats detected in the analyzed data
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Detailed results
                                st.subheader("📋 Detailed Results")
                                
                                # Create results dataframe
                                results_df = predict_df.copy()
                                results_df['Threat_Prediction'] = predictions
                                results_df['Threat_Probability'] = probabilities.flatten()
                                results_df['Risk_Level'] = results_df['Threat_Probability'].apply(
                                    lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
                                )
                                
                                # Filter options
                                col1, col2 = st.columns(2)
                                with col1:
                                    show_filter = st.selectbox("Show records:", 
                                                             ["All", "Threats Only", "Safe Only"])
                                with col2:
                                    sort_by = st.selectbox("Sort by:", 
                                                         ["Threat Probability", "Original Order"])
                                
                                # Apply filters
                                filtered_df = results_df.copy()
                                if show_filter == "Threats Only":
                                    filtered_df = filtered_df[filtered_df['Threat_Prediction'] == 1]
                                elif show_filter == "Safe Only":
                                    filtered_df = filtered_df[filtered_df['Threat_Prediction'] == 0]
                                
                                if sort_by == "Threat Probability":
                                    filtered_df = filtered_df.sort_values('Threat_Probability', ascending=False)
                                
                                # Display table
                                st.dataframe(filtered_df, use_container_width=True, height=400)
                                
                                # Download results
                                csv = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name=f"threat_analysis_{int(time.time())}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error(probabilities)  # Error message
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("📈 Model Analytics")
        
        if st.session_state.training_history:
            history = st.session_state.training_history
            
            # Training curves
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training & Validation Loss', 'Validation Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Loss plot
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['train_loss'], 
                          name='Training Loss', line=dict(color='#ff6b6b')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['val_loss'], 
                          name='Validation Loss', line=dict(color='#51cf66')),
                row=1, col=1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['val_accuracy'], 
                          name='Validation Accuracy', line=dict(color='#339af0')),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
            with col2:
                st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
            with col3:
                st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.2%}")
                
        else:
            st.info("📊 Train a model first to see analytics.")
    
    with tab4:
        st.header("🧪 Single Prediction")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first.")
        else:
            st.subheader("Enter system log values:")
            
            # Input form
            col1, col2 = st.columns(2)
            
            with col1:
                process_id = st.number_input("Process ID", value=1000)
                thread_id = st.number_input("Thread ID", value=2000)
                parent_process_id = st.number_input("Parent Process ID", value=500)
                user_id = st.number_input("User ID", value=1001)
            
            with col2:
                mount_namespace = st.number_input("Mount Namespace", value=4026531840)
                args_num = st.number_input("Arguments Number", value=3)
                return_value = st.number_input("Return Value", value=0)
            
            if st.button("🔍 Analyze Single Record", type="primary"):
                # Create input data
                input_data = pd.DataFrame({
                    'processId': [process_id],
                    'threadId': [thread_id],
                    'parentProcessId': [parent_process_id],
                    'userId': [user_id],
                    'mountNamespace': [mount_namespace],
                    'argsNum': [args_num],
                    'returnValue': [return_value]
                })
                
                predictions, probabilities = predict_threats(input_data)
                
                if predictions is not None:
                    prediction = predictions[0]
                    probability = probabilities[0][0]
                    
                    # Display result
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.markdown("""
                            <div class="threat-alert">
                                🚨 <strong>THREAT DETECTED!</strong><br>
                                This record is classified as suspicious
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="safe-alert">
                                ✅ <strong>RECORD IS SAFE</strong><br>
                                No threat detected in this record
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Gauge chart for probability
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = probability * 100,
                            title = {'text': "Threat Probability (%)"},
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
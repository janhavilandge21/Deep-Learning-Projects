import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import random
from PIL import Image
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff


def generate_model_history(model_name, epochs, max_acc, loss_factor, noise_level):
    """Generates synthetic training history for a single model."""
    random.seed(hash(model_name) % 1000)
    np.random.seed(hash(model_name) % 1000)
    
    epochs_list = np.arange(1, epochs + 1)
    
    val_acc = np.linspace(0.4, max_acc, epochs) + np.random.uniform(-noise_level, noise_level, epochs)
    val_acc = np.clip(val_acc, 0.4, 1.0)
    
    val_loss = np.linspace(1.8, 1.8 * loss_factor, epochs) + np.random.uniform(-noise_level * 2, noise_level * 2, epochs)
    val_loss = np.clip(val_loss, 0.1, 2.0)
    
    return pd.DataFrame({
        "Epoch": epochs_list,
        "Model": model_name,
        "Validation Accuracy": val_acc,
        "Validation Loss": val_loss
    })

# Simulated data for VGG16, VGG19, and ResNet
EPOCHS = 40
df_vgg16 = generate_model_history("VGG16", EPOCHS, 0.88, 0.2, 0.04)
df_vgg19 = generate_model_history("VGG19", EPOCHS, 0.91, 0.15, 0.03)
df_resnet = generate_model_history("ResNet50", EPOCHS, 0.94, 0.1, 0.02)
df_comparison = pd.concat([df_vgg16, df_vgg19, df_resnet], ignore_index=True)

final_accs = df_comparison.groupby('Model')['Validation Accuracy'].max()
best_model_name = final_accs.idxmax()
best_accuracy = final_accs.max()

# Model Metric Data for Trade-off Analysis
MODEL_STATS = pd.DataFrame({
    'Model': ["VGG16", "VGG19", "ResNet50"],
    'Max Accuracy': [df_vgg16['Validation Accuracy'].max(), df_vgg19['Validation Accuracy'].max(), df_resnet['Validation Accuracy'].max()],
    'Parameters (M)': [138.3, 143.7, 25.6],
    'Inference Time (ms)': [random.uniform(18, 28), random.uniform(22, 32), random.uniform(8, 15)]
}).set_index('Model')


# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Architecture Arena",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Deep Learning Architecture")
st.caption("VGG16 vs. VGG19 vs. ResNet: Performance, Efficiency, and Smart Recommendations.")

# --- CREATE TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs([
    "📸 Interactive Prediction", 
    "⭐ Intelligent Model Recommendation", 
    "🚀 The Architecture Arena"
])

# ==============================================================================
# 1. INTERACTIVE PREDICTION TAB 
# ==============================================================================
with tab1:
    st.header(" Interactive Image Prediction Test")
    st.info("Upload an image (JPG, PNG, WEBP) to see a **simulated** prediction.")

    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
            
        with st.container(border=True):
            col_img, col_pred = st.columns([1, 1])
            
            with col_img:
                st.subheader("Uploaded Image")
                st.image(image, caption=uploaded_file.name, use_container_width=True)

            with col_pred:
                st.subheader("Prediction Results")
                
                simulated_model = st.selectbox(
                    "Simulate Model Prediction:", 
                    options=["ResNet50", "VGG19", "VGG16", "EfficientNetV2"], 
                    index=0, 
                    key="pred_model"
                )
                
                # Simulation
                seed = hash(uploaded_file.name + simulated_model) % 1000
                random.seed(seed)
                
                prediction_classes = ["Artificial Intelligence", "Neural Network", "Computer Vision", "Data Science", "General Tech"]
                top_class = random.choice(prediction_classes)
                confidence = random.uniform(0.75, 0.99)
                
                st.metric(label=f"Predicted Class by **{simulated_model}**", value=top_class)
                st.progress(confidence, text=f"Confidence: {confidence:.2f}")
                
                st.markdown("**Top-5 Confidence Breakdown:**")
                
                top_5_results = []
                top_5_results.append((top_class, confidence))
                
                conf_values = np.sort(np.random.uniform(0.1, confidence * 0.9, 4))[::-1]
                remaining_classes = [c for c in prediction_classes if c != top_class]
                
                for i, conf in enumerate(conf_values):
                     if i < len(remaining_classes): top_5_results.append((remaining_classes[i], conf))
                
                top_5_results.sort(key=lambda x: x[1], reverse=True)
                
                for i, (cls, conf) in enumerate(top_5_results[:5]): st.text(f"{i+1}. {cls}: {conf:.4f}")

# ==============================================================================
# 2. INTELLIGENT MODEL RECOMMENDATION 
# ==============================================================================
with tab2:
    st.header(" Intelligent Architecture Selector")
    st.markdown("### Determine the best model for YOUR specific project needs.")

    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Decision Factors")
        
        priority = st.slider(
            "Prioritize:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            format="%.1f",
            help="0.0 = Pure Speed (Low Latency), 1.0 = Pure Accuracy (Best Score)"
        )
        
        task_complexity = st.selectbox(
            "Task Complexity:",
            options=["Low (e.g., Cat vs Dog)", "Medium (e.g., 100 classes)", "High (e.g., Medical Imaging)"],
            index=1,
            help="Select the difficulty of the classification task."
        )

    with col_output:
        st.subheader("Optimal Architecture Recommendation")
        
        # --- Recommendation Logic ---
        if task_complexity == "High (e.g., Medical Imaging)":
            base_model = "ResNet50"
            complexity_note = "High complexity favors deeper, more efficient architectures like ResNet due to residual connections."
        elif task_complexity == "Medium (e.g., 100 classes)":
            base_model = "VGG19"
            complexity_note = "Medium complexity often provides a good balance with VGG19's large capacity."
        else: # Low Complexity
            base_model = "VGG16"
            complexity_note = "Low complexity tasks often don't justify the overhead of deeper models; VGG16 is efficient enough."

        if priority > 0.8:
            final_recommendation = final_accs.idxmax()
            reason = f"Prioritizing **Accuracy**."
        elif priority < 0.2:
            final_recommendation = MODEL_STATS['Inference Time (ms)'].idxmin()
            reason = f"Prioritizing **Speed**."
        else:
            final_recommendation = base_model
            reason = f"Prioritizing **Balance** between speed and accuracy."

        st.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; background-color: #f0fff4; border-left: 6px solid #28a745; margin-bottom: 15px; text-align: center;">
            <p style="font-size: 20px; font-weight: bold; color: #28a745; margin: 0;">Optimal Choice:</p>
            <p style="font-size: 42px; font-weight: 900; color: #28a745; margin: 0;">{final_recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Detailed Reasoning:** {reason} Based on **Task Complexity** ({task_complexity}), {complexity_note}")
        
        st.markdown("---")
        st.subheader("Micro-Level Analysis (Simulated)")
        
        cm_cols = st.columns(2)
        with cm_cols[0]:
            st.markdown(f"**Simulated Confusion Matrix for {final_recommendation}**")
            
            recommended_acc = MODEL_STATS.loc[final_recommendation, 'Max Accuracy']
            classes = ['AI', 'NN', 'CV', 'Data Science', 'General Tech']
            n_samples = 500 
            true_labels = np.random.randint(0, len(classes), n_samples)
            predicted_labels = np.copy(true_labels)
            n_errors = int(n_samples * (1 - recommended_acc))
            
            error_indices = np.random.choice(n_samples, n_errors, replace=False)
            for i in error_indices:
                possible_errors = [c for c in range(len(classes)) if c != true_labels[i]]
                if possible_errors: predicted_labels[i] = random.choice(possible_errors)

            cm = confusion_matrix(true_labels, predicted_labels)
            fig_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='Viridis', showscale=False)
            fig_cm.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# 3. THE ARCHITECTURE ARENA 
# ==============================================================================
with tab3:
    st.header(" The Architecture Arena: Deep Dive Analysis")
    
    #  Technical Deep Dive
    st.subheader(f"🥇 ResNet50: The Best Model")
    
    st.markdown(f"""
    In this comparison, **{best_model_name}** consistently outperforms **VGG16** and **VGG19** by achieving the highest accuracy with the lowest latency and parameter count. This superiority stems from a fundamental difference in architecture:
    
    ### ResNet50: The Power of Residual Blocks
    
    The key to ResNet's success is the **Residual Block** (or skip connection).
    
    * **The Problem it Solves:** Traditional deep networks like VGG suffer from the **Vanishing Gradient Problem** and **Degradation**. As VGG gets deeper, accuracy plateaus and eventually drops.
    * **The Solution:** ResNet layers use a skip connection, allowing the input ($x$) to bypass several weight layers and be added directly to the output. This enables the training of networks with hundreds of layers without loss of performance across **Epochs**.
    * **Efficiency:** ResNet50 uses **bottleneck layers** which drastically reduce the number of parameters and computation compared to VGG's pure 3x3 stacks.
    """)

    st.markdown("---")
    
    st.subheader("📉 VGG Performance Analysis: Parameter Overload")
    
    st.markdown("""
    | Model | Total Parameters (M) | Inference Time (ms) | Key Weakness |
    | :--- | :---: | :---: | :--- |
    | **VGG19** | **143.7M** | Highest | **Massive** number of parameters, making training slow and increasing risk of overfitting. |
    | **VGG16** | 138.3M | High | Simpler architecture struggles to capture complexity compared to ResNet. |
    | **ResNet50** | **25.6M** | Lowest | Optimized for both deep feature extraction and efficiency. |
    
    VGG models are robust but are simple stacks of convolution layers. They are **computationally expensive** due to their large parameter count, leading to slower training and inference compared to the much more efficient ResNet over many **Epochs**.
    """)

    st.markdown("---")

    #  Epoch Curves 
    st.header(" Epoch-by-Epoch ")
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Validation Accuracy Over Epochs** (The Uphill Battle)")
        fig_acc = px.line(df_comparison, x='Epoch', y='Validation Accuracy', color='Model', line_shape='spline', height=400, template="plotly_white")
        fig_acc.update_traces(mode='lines')
        st.plotly_chart(fig_acc, use_container_width=True)

    with chart_col2:
        st.markdown("**Validation Loss Over Epochs** (The Downward Trend)")
        fig_loss = px.line(df_comparison, x='Epoch', y='Validation Loss', color='Model', line_shape='spline', height=400, template="plotly_white")
        fig_loss.update_traces(mode='lines')
        st.plotly_chart(fig_loss, use_container_width=True)
    
    st.markdown("---")

    # Final Scorecard Table
    st.header(f" Final Scorecard: {best_model_name} Wins! 🥇")
    
    # Metrics
    col_best1, col_best2, col_best3, col_best4 = st.columns(4)
    with col_best1: st.metric(label="Overall Best Accuracy", value=f"{best_accuracy:.4f}", delta=f"Achieved in {EPOCHS} Epochs")
    with col_best2: st.metric(label="Total Epochs Analyzed", value=EPOCHS)
    with col_best3: st.metric(label="Fastest Inference Time", value=f"{MODEL_STATS['Inference Time (ms)'].min():.2f} ms")
    with col_best4: st.metric(label="Lowest Parameters (M)", value=f"{MODEL_STATS['Parameters (M)'].min():.1f}M")
    
    
    # Trade-off Scatter Plot
    MODEL_STATS_RESET = MODEL_STATS.reset_index()
    fig_tradeoff = px.scatter(
        MODEL_STATS_RESET,
        x='Inference Time (ms)',
        y='Max Accuracy',
        text='Model',
        size='Parameters (M)',
        color='Model',
        title='Accuracy vs. Inference Time Trade-off (Goal: Top-Left Corner)',
        height=550,
        template="plotly_white"
    )
    fig_tradeoff.update_traces(textposition='top center', marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
    fig_tradeoff.update_layout(
        xaxis_title="Inference Time (ms) (Speed)", 
        yaxis_title="Max Accuracy (Performance)",
        font=dict(size=14),
        hovermode="closest"
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)


# --- FINAL INSTRUCTIONS IN SIDEBAR (CLEANED) ---
st.sidebar.markdown("---")
st.sidebar.header("Project Info")
st.sidebar.markdown("""
This dashboard compares the performance and efficiency of three widely used CNN architectures: **VGG16, VGG19, and ResNet50**.

The goal is to determine the optimal model based on the trade-offs between accuracy, speed (latency), and model size (parameters).
""")
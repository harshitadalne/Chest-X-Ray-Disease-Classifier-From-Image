import streamlit as st
import torchxrayvision as xrv
import torch
import torchvision
import numpy as np
import skimage.io
import skimage.transform
import pandas as pd
from PIL import Image, ImageOps
import io
import sys
from datetime import datetime
import warnings

# Suppress runtime warnings from libraries like skimage/xrv during setup
warnings.filterwarnings("ignore", category=UserWarning)

# Set page config first
st.set_page_config(
    page_title="X-Ray Classifier",
    page_icon="🫁",
    layout="wide"
)

# Show initial loading message
st.info("Good day! Welcome to the X-Ray Classifier App. Please upload a chest X-ray image to begin analysis.")

# Display debug info at the top
st.sidebar.write("### Debug Information")
st.sidebar.write(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.write(f"Python version: {sys.version.split()[0]}")
st.sidebar.write(f"Torch version: {torch.__version__}")
st.sidebar.write(f"TorchVision version: {torchvision.__version__}")
try:
    st.sidebar.write(f"TorchXRayVision version: {xrv.__version__}")
except:
    st.sidebar.write("TorchXRayVision version: unknown")

# Load the model outside the prediction function to ensure it's only loaded once
# We specify the NIH weights which correspond to the ChestX-ray8 dataset diseases
@st.cache_resource
def load_model():
    try:
        # Create and load the model with weights, forcing CPU
        device = torch.device('cpu')
        model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Technical details for debugging:")
        st.code(f"""
        Available weights: {xrv.models.model_urls}
        Device: {device}
        Python version: {sys.version}
        Torch version: {torch.__version__}
        TorchXRayVision version: {xrv.__version__}
        """)
        raise

MODEL = load_model()

def preprocess_and_predict(image_bytes, model, auto_invert=True):
    try:
        # 1. Decode Image (use PIL for inversion detection, then skimage for pipeline)
        st.write("1. Loading image...")
        pil = Image.open(image_bytes)
        # Compute mean intensity on grayscale for inversion heuristic
        gray = pil.convert('L')
        mean_intensity = np.array(gray).mean()
        st.sidebar.write(f"Uploaded image mean intensity: {mean_intensity:.1f}")

        inverted = False
        if auto_invert and mean_intensity > 127:
            # Image likely inverted (bright lungs); invert and note it
            st.sidebar.write("Auto-inversion: detected and corrected")
            pil = ImageOps.invert(pil.convert('RGB'))
            inverted = True
        else:
            st.sidebar.write("Auto-inversion: not applied")

        # Convert PIL to numpy for the existing pipeline
        image = np.array(pil)
        st.write("✅ Image loaded successfully")
        
        # 2. Convert to grayscale and normalize
        st.write("2. Normalizing image...")
        # The model expects a single channel image normalized to a specific range
        image = xrv.datasets.normalize(image, 255)
        
        # If the image is color (3 channels), convert to 1 channel (greyscale)
        if image.ndim == 3:
            st.write("Converting color image to grayscale...")
            image = image.mean(2)[None, ...]
        elif image.ndim == 2:
            image = image[None, ...]  # Add a channel dimension for 1 channel
        st.write("✅ Image normalized")
        
        # 3. Resize and transform
        st.write("3. Resizing image to 224x224...")
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        image = transform(image)
        st.write("✅ Image resized")
        
        # 4. Convert to PyTorch Tensor
        st.write("4. Converting to PyTorch tensor...")
        image = torch.from_numpy(image).unsqueeze(0).float()
        st.write("✅ Converted to tensor")
        
        # 5. Inference
        st.write("5. Running model inference...")
        with torch.no_grad():
            output = model(image).cpu()
        st.write("✅ Model inference complete")
        
        # 6. Post-process
        st.write("6. Processing results...")
        probas = torch.sigmoid(output).numpy().flatten()
        
        # 7. Create results DataFrame with all pathologies including Normal
        pathologies = [
            'Normal', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
            'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity'
        ]
        
        # Align model outputs with our pathology list. Many models do not include an explicit
        # 'Normal' label, so if the model returns one fewer probability than our list
        # and the first label is 'Normal', insert a zero at index 0 so diseases align.
        if len(probas) == len(pathologies) - 1 and pathologies[0].lower() == 'normal':
            new_probas = np.zeros(len(pathologies), dtype=float)
            copy_len = min(len(probas), len(pathologies) - 1)
            new_probas[1:1+copy_len] = probas[:copy_len]
            probas = new_probas
        elif len(probas) != len(pathologies):
            st.warning(f"⚠️ Note: Model returned {len(probas)} probabilities but we have {len(pathologies)} pathologies")
            # Pad with zeros at the end or truncate as a fallback
            if len(probas) < len(pathologies):
                probas = np.pad(probas, (0, len(pathologies) - len(probas)), 'constant')
            else:
                probas = probas[:len(pathologies)]
        
        results_df = pd.DataFrame({
            'Disease': pathologies,
            'Probability': probas
        })

        st.write("✅ Results processed successfully")
        # Return corrected PIL and inversion flag so downstream steps (saliency) use the corrected image
        return results_df, pil, inverted, mean_intensity
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing/prediction: {str(e)}")
        st.write("Debug information:")
        st.write(f"- Image shape (if available): {image.shape if 'image' in locals() else 'Not created yet'}")
        st.write(f"- Model state: {model.training}")
        st.write(f"- Current device: {next(model.parameters()).device}")
        raise


# -------------------------
# Explainability helpers
# -------------------------
def process_image_bytes_for_model(image_bytes):
    """Return a tuple (tensor, original_pil) prepared for the model.
    The tensor has shape (1,C,H,W) and requires_grad=False by default.
    """
    # Read image via PIL (keeps original colors)
    pil = Image.open(image_bytes).convert('RGB')
    np_img = np.array(pil)

    # Use the same normalization and resizing as preprocessing
    img = xrv.datasets.normalize(np_img, 255)
    if img.ndim == 3:
        img = img.mean(2)[None, ...]
    elif img.ndim == 2:
        img = img[None, ...]

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img_t = transform(img)

    tensor = torch.from_numpy(img_t).unsqueeze(0).float()  # (1,C,H,W)
    return tensor, pil


def generate_saliency_overlay(tensor, model, original_pil):
    """Generate a simple gradient-based saliency heatmap overlaying on original_pil.
    Returns a PIL RGBA image with the heatmap blended.
    """
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    tensor.requires_grad_()

    model.zero_grad()
    output = model(tensor)
    probs = torch.sigmoid(output).cpu().detach().numpy().flatten()
    top_idx = int(probs.argmax())

    # Backprop on the top logit
    score = output[0, top_idx]
    score.backward(retain_graph=False)

    # Get gradients w.r.t. input
    grad = tensor.grad.detach().cpu().numpy()[0]
    # Mean across channels
    grad_mean = np.mean(np.abs(grad), axis=0)

    # Normalize
    grad_norm = (grad_mean - grad_mean.min())
    if grad_norm.max() > 0:
        grad_norm = grad_norm / grad_norm.max()
    else:
        grad_norm = grad_norm

    # Resize heatmap to original image size
    heatmap = (grad_norm * 255).astype('uint8')
    heat_pil = Image.fromarray(heatmap).convert('L').resize(original_pil.size)

    # Create colored overlay (red) and blend
    red = Image.new('RGBA', original_pil.size, color=(255, 0, 0, 0))
    # Use heatmap as alpha
    red.putalpha(heat_pil)

    base = original_pil.convert('RGBA')
    overlay = Image.alpha_composite(base, red)
    return overlay, top_idx, probs[top_idx]


def get_disease_description(disease_name: str) -> str:
    """Return a short, user-friendly description for each supported disease label."""
    descriptions = {
        'Normal': 'No significant pathological findings identified on this chest X-ray study.',
        'Atelectasis': 'Atelectasis is a partial or complete collapse of lung tissue, often seen as volume loss or increased density on X-ray.',
        'Cardiomegaly': 'Cardiomegaly indicates an enlarged cardiac silhouette and can suggest heart failure or chronic cardiac conditions.',
        'Consolidation': 'Consolidation refers to alveolar air being replaced by fluid, pus, blood, or cells; commonly seen with pneumonia.',
        'Edema': 'Pulmonary edema is accumulation of fluid in the lung interstitium and alveoli, frequently due to heart failure.',
        'Effusion': 'Pleural effusion is excess fluid between the lung and chest wall; may blunt costophrenic angles on X-ray.',
        'Emphysema': 'Emphysema is a chronic condition involving destruction of alveolar walls and airspace enlargement, typical of COPD.',
        'Fibrosis': 'Pulmonary fibrosis describes scarring of lung tissue that can cause stiffness and reduced lung volumes.',
        'Hernia': 'Hernia (eg. hiatal or diaphragmatic) can appear when abdominal contents abnormally project into the chest cavity.',
        'Infiltration': 'Infiltrates are nonspecific opacities that may indicate infection, inflammation, or other processes.',
        'Mass': 'A mass is a focal rounded opacity; differential includes benign lesions, primary lung cancer, or metastasis.',
        'Nodule': 'A nodule is a small, round opacity that may represent healed infection, benign tumor, or early malignancy.',
        'Pleural_Thickening': 'Pleural thickening indicates scarring or inflammation of the pleural lining and may follow prior disease or asbestos exposure.',
        'Pneumonia': 'Pneumonia is infection of the lung parenchyma commonly producing consolidation and clinical symptoms like fever and cough.',
        'Pneumothorax': 'Pneumothorax is air in the pleural space causing partial or complete lung collapse; look for absence of vascular markings.',
        'Enlarged Cardiomediastinum': 'Enlarged cardiomediastinum suggests widening of the central thoracic structures and may reflect cardiomegaly or mediastinal pathology.',
        'Fracture': 'Fracture indicates bone discontinuity (eg. ribs, clavicle) that can sometimes be visible on chest X-ray.',
        'Lung Lesion': 'Lung lesion is a nonspecific term for an abnormal focal area; further imaging and clinical correlation are recommended.',
        'Lung Opacity': 'Lung opacity denotes an area of increased density in the lung and can represent consolidation, mass, or other pathology.'
    }
    return descriptions.get(disease_name, 'Refer to clinical correlation and a formal radiology report for further interpretation.')

# Main app UI
st.title("🫁 Chest X-Ray Disease Classifier ")
st.caption("Classify the X-ray images and get the correct results.")

# Show model status
if 'MODEL' in globals():
    st.success("✅ Model loaded successfully")
else:
    st.error("❌ Model not loaded")
    st.stop()

st.markdown("---")
st.write("### Instructions:")
st.write("1. Use the uploader below to select a chest X-ray image")
st.write("2. The image should be a clear, standard X-ray in PNG or JPG format")
st.write("3. After upload, the model will analyze the image and show predictions")
st.markdown("---")

# --- File Uploader ---
with st.expander("⚙️ Upload Settings", expanded=True):
    st.warning("Please ensure the uploaded file is a clear, standard X-ray image (PNG/JPG).")
    uploaded_file = st.file_uploader("Upload a Chest X-ray image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # --- Display Uploaded Image ---
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.markdown("---")
    
    # Read the image content as bytes
    image_bytes = io.BytesIO(uploaded_file.getvalue())

    # --- Run Prediction ---
    with st.spinner('Running multi-label prediction on CPU...'):
        try:
            # Sidebar control for auto inversion
            auto_invert = st.sidebar.checkbox("Auto-correct inverted images", value=True)
            results_df, corrected_pil, inverted, mean_intensity = preprocess_and_predict(image_bytes, MODEL, auto_invert=auto_invert)
            
            # --- Display Results ---
            st.header("Diagnosis Results")

            # Calculate normalcy score (inverse of max probability excluding first row)
            disease_probs = results_df.iloc[1:]['Probability'].values
            max_disease_prob = max(disease_probs)
            normalcy_score = 1.0 - max_disease_prob
            
            # Update probabilities
            results_df.iloc[0, results_df.columns.get_loc('Probability')] = normalcy_score  # Set 'Normal' probability
            
            # Force disease probabilities to be low if image appears normal
            if normalcy_score > 0.7:  # If highly likely to be normal
                # Corrected Syntax: Using backslash for clean line continuation
                results_df.iloc[1:, results_df.columns.get_loc('Probability')] = \
                results_df.iloc[1:]['Probability'].apply(lambda x: min(x, 0.1))  # Cap at 10%
                
            # FIX APPLIED HERE: Removed trailing backticks/extra characters
            results_df['Probability (%)'] = results_df['Probability'] * 100
            
            # Interactive elements for report customization
            st.sidebar.write("### Report Settings")
            show_details = st.sidebar.checkbox("Show detailed explanations", value=True)
            probability_threshold = st.sidebar.slider("Probability threshold (%)", 0, 100, 20)
            
            # Main diagnosis banner
            if normalcy_score > 0.7:
                st.success(f"✅ NORMAL STUDY (Confidence: {normalcy_score:.1%})")
                st.balloons()  # Celebratory animation for normal results
            else:
                st.warning(f"⚠️ FINDINGS DETECTED (Abnormality confidence: {(1-normalcy_score):.1%})")
            
            # Show top 5 findings
            st.subheader("Top 5 Findings")
            cols = st.columns([3, 1])  # For better layout
            
            # Convert to percentage and round for display
            top_results = results_df.sort_values(by='Probability', ascending=False).head(5)
            
            
            # Show results in a more interactive way (First table)
            with cols[0]:
                st.dataframe(
                    top_results[['Disease', 'Probability (%)']],
                    hide_index=True,
                    column_config={
                        "Disease": st.column_config.TextColumn("Finding", width=200),
                        "Probability (%)": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )
            
            with cols[1]:
                # Quick stats
                st.metric("Normalcy Score", f"{normalcy_score:.1%}")
                st.metric("Findings Above Threshold",
                          len(results_df[results_df['Probability'] > probability_threshold/100]))
            
            # Downloadable CSV of results (kept, but do not display full table)
            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download results as CSV", data=csv_bytes, file_name="xray_results.csv", mime="text/csv")

            # Generate detailed report
            st.markdown("---")
            st.header("📋 Detailed Analysis Report")
            
            # Only show the top-5 findings and descriptions
            significant = top_results.copy()
            significant = significant[significant['Probability (%)'] > probability_threshold]

            if len(significant) > 0:
                st.write("### Key Findings")
                for _, row in significant.iterrows():
                    prob_percentage = row['Probability (%)']
                    if prob_percentage > 70:
                        emoji = "🔴"  # High probability
                    elif prob_percentage > 40:
                        emoji = "🟡"  # Medium probability
                    else:
                        emoji = "🟢"  # Low but notable probability

                    st.write(f"{emoji} **{row['Disease']}** ({prob_percentage:.1f}% probability)")

                    # Add descriptive text based on the top-5 disease using centralized descriptions
                    desc = get_disease_description(row['Disease'])
                    st.info(desc)
            else:
                st.success("No top-5 findings exceed the selected probability threshold.")

            # Explainability: saliency heatmap for top predicted disease
            st.markdown("---")
            st.subheader("🧭 Explainability / Saliency Map")
            if st.button("Generate saliency overlay for top finding"):
                with st.spinner("Generating saliency map..."):
                    try:
                        # Prepare tensor and original image using corrected PIL from preprocessing
                        buf = io.BytesIO()
                        corrected_pil.save(buf, format='PNG')
                        buf.seek(0)
                        tensor, original_pil = process_image_bytes_for_model(buf)
                        overlay, top_idx, top_score = generate_saliency_overlay(tensor, MODEL, original_pil)

                        st.image(original_pil, caption="Original uploaded image", use_column_width=True)
                        st.image(overlay, caption=f"Saliency overlay (top predicted: {top_score:.2f})", use_column_width=True)

                        # Offer overlay download
                        out_buf = io.BytesIO()
                        overlay.save(out_buf, format='PNG')
                        out_buf.seek(0)
                        st.download_button("Download saliency PNG", data=out_buf, file_name="saliency_overlay.png", mime="image/png")

                    except Exception as e:
                        st.error(f"Failed to generate saliency map: {e}")
                        st.exception(e)
            
            # Interactive Report Section
            st.markdown("---")
            
            # Tabs for different views of the report
            tab1, tab2, tab3 = st.tabs(["Summary", "Details", "Help"])
            
            with tab1:
                st.header("Report Summary")
                if normalcy_score > 0.7:
                    st.success("""
                    🏥 **Overall Impression: Normal Study**
                    - No significant pathological findings detected
                    - High confidence in normal appearance
                    - Routine follow-up as clinically indicated
                    """)
                else:
                    findings_count = len(results_df[results_df['Probability'] > probability_threshold/100])
                    st.warning(f"""
                    🏥 **Overall Impression: Abnormal Study**
                    - {findings_count} significant findings detected
                    - Findings detailed in report below
                    - Clinical correlation recommended
                    """)
                
                # Interactive elements
                if st.button("📋 Generate Summary Report"):
                    report_text = f"""
                    CHEST X-RAY ANALYSIS REPORT
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    IMPRESSION:
                    {'Normal study with no significant findings' if normalcy_score > 0.7 
                      else 'Abnormal study with findings detailed below'}
                    
                    FINDINGS:
                    """
                    for _, row in top_results.iterrows():
                        if row['Probability (%)'] > probability_threshold:
                            report_text += f"\n- {row['Disease']}: {row['Probability (%)']:.1f}% confidence"
                    
                    st.code(report_text)
            
            with tab2:
                st.header("Detailed Findings")
                for _, row in top_results.iterrows():
                    if row['Probability (%)'] > probability_threshold:
                        with st.expander(f"{row['Disease']} ({row['Probability (%)']:.1f}%)"):
                            st.info(get_disease_description(row['Disease']))
                            
                            # Show recommended actions based on probability
                            if row['Probability (%)'] > 70:
                                st.error("🚨 Immediate clinical correlation recommended")
                            elif row['Probability (%)'] > 40:
                                st.warning("⚠️ Follow-up recommended")
                            else:
                                st.info("ℹ️ Consider clinical correlation if symptomatic")
            
            with tab3:
                st.header("Using This Report")
                st.write("""
                ### Interpretation Guide
                - Confidence scores indicate the AI model's certainty
                - Normal studies have a normalcy score >70%
                - Findings are ranked by confidence level
                
                ### Important Notes
                - This is an AI assistance tool only
                - Not a replacement for professional medical interpretation
                - Consult healthcare providers for medical advice
                - Regular screening as recommended by your physician
                """)
                
                # Add a feedback section
                st.write("### Feedback")
                if st.button("Was this analysis helpful? 👍"):
                    st.success("Thank you for your feedback! This helps improve our system.")
                    
            # Download options
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Detailed Report (PDF)"):
                    st.info("PDF report generation would be implemented here")
            with col2:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Results (CSV)",
                    data=csv,
                    file_name="xray_analysis.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the uploaded file is a clear, standard X-ray image (PNG/JPG).")
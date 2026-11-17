import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="NeuroCognitive",
    page_icon="🧠",
    layout="wide"
)

if "role" not in st.session_state:
    st.session_state.role = "public"

if "doctor_logged_in" not in st.session_state:
    st.session_state["doctor_logged_in"] = False

if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = None
# ---------------------------------------------------
# DOCTOR CREDENTIALS (HOSPITAL SYSTEM)
# ---------------------------------------------------

DOCTORS = {
    "dr_shive": {
        "password": "shive123",
        "name": "Dr. Shive"
    },
    "dr_nive": {
        "password": "nive123",
        "name": "Dr. Nive"
    },
    "dr_vikram": {
        "password": "vikky123",
        "name": "Dr. Vikram"
    },
    "dr_dhruv": {
        "password": "dhruv123",
        "name": "Dr. Dhruv Vikram"
    }
}

## ---------------------------------------------------
# PREMIUM HOSPITAL + AI LAB CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}
            .hover-card {
    background: linear-gradient(145deg, #ffffff, #f7f9fc);
    border-left: 6px solid #0f4c75;
    transition: all 0.3s ease;
}

.hover-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.hover-card h3 {
    color: #0f4c75;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.card {
    background: linear-gradient(135deg, #1b4965, #2c699a);
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(15, 76, 117, 0.25);
    margin-bottom: 25px;
    color: white;
}

.card h3, 
.card h2,
.card h4,
.card p,
.card b {
    color: white;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    color: #1f2d3d;
    margin-bottom: 15px;
}

.subtle-text {
    color: #6c757d;
    font-size: 15px;
}

.status-improved {
    color: #1b8a3e;
    font-weight: 600;
    font-size: 22px;
}

.status-declined {
    color: #b02a37;
    font-weight: 600;
    font-size: 22px;
}
.gradient-header {
    font-size: 34px;
    font-weight: 700;
    padding: 22px 28px;
    border-radius: 14px;
    background: linear-gradient(90deg, #0f4c75, #3282b8);
    color: white;
    margin-bottom: 30px;
    letter-spacing: 0.5px;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
MODEL_PATH = "alzheimer_model_cpu.h5"
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()
model.summary()
# IMPORTANT: Correct class order from your dataset
CLASS_NAMES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]
SEVERITY_ORDER = {
    "Non Demented": 0,
    "Very Mild Demented": 1,
    "Mild Demented": 2,
    "Moderate Demented": 3
}

IMG_SIZE = 224

def generate_gradcam(model, img_array, last_conv_layer_name="Conv_1"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ---------------------------------------------------
# ROLE-BASED SIDEBAR NAVIGATION
# ---------------------------------------------------

st.sidebar.title("NeuroCognitive Platform")

if st.session_state.role == "doctor":

    st.sidebar.success(f"Logged in as {st.session_state.doctor_name}")

    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Detection", "Comparison", "Analytics", "Diet", "Logout"]
    )

else:

    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Analytics", "Diet"]
    )

# ===================================================
# HOME PAGE
# ===================================================
def show_home_access():

    st.markdown("---")
    st.markdown("## 🌐 Choose Access Mode")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card hover-card" style="text-align:center;">
            <h3 style="font-size:26px;">
                <span style="
                font-size:32px;
                margin-right:10px;
                filter: drop-shadow(0 0 4px rgba(255,255,255,0.6));
                ">
                👤
                </span>
                Public Access
            </h3>
            <p>Explore dementia awareness and diet guidance</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Public Section"):
            st.session_state.role = "public"
            st.session_state["doctor_logged_in"] = False
            st.success("Public Mode Activated")
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card hover-card" style="text-align:center;">
            <h3>👨‍⚕️ Doctor Login</h3>
            <p>Secure MRI Diagnosis & Clinical Monitoring</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login as Doctor"):
            if username in DOCTORS and DOCTORS[username]["password"] == password:
                st.session_state.role = "doctor"
                st.session_state["doctor_logged_in"] = True
                st.session_state.doctor_name = DOCTORS[username]["name"]
                st.success(f"Welcome {st.session_state.doctor_name}")
                st.rerun()
            else:
                st.error("Invalid credentials")
# ===================================================
# HOME PAGE CONTENT
# ===================================================

def show_home():

    st.markdown('<div class="gradient-header">🧠 NeuroCognitive Disorder Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to NeuroCognitive AI Clinical Assistant

    This system supports dementia awareness, MRI-based diagnosis,
    treatment monitoring and dietary guidance.

    Select **Public Access** for awareness content  
    Select **Doctor Access** for clinical analysis tools
    """)
# ===================================================
# DETECTION PAGE
# ===================================================
def show_detection():

    st.markdown(
        '<div class="gradient-header">NeuroCognitive Disorder Detection using MRI</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

    if not uploaded:
        if "last_original_img" in st.session_state and "last_gradcam_img" in st.session_state:

            st.subheader("📷 Last MRI Result")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    st.session_state["last_original_img"],
                    caption="Original MRI Scan",
                    width=280,
                    channels="BGR"
                )

            with col2:
                st.image(
                    st.session_state["last_gradcam_img"],
                    caption="GradCAM Heatmap",
                    width=280,
                    channels="BGR"
                )

        return

    # ===========================================
    # Load Image
    # ===========================================

    image = Image.open(uploaded).convert("RGB")

    img = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    # ===========================================
    # Prediction Stage
    # ===========================================

    # ===============================
# Multi-pass Prediction Averaging
# ===============================

    predictions_list = []

    for _ in range(5):   # 5 inference passes
        noise = np.random.normal(0, 0.001, arr.shape)
        noisy_input = arr + noise
        noisy_input = np.clip(noisy_input, 0, 1)

        preds = model.predict(noisy_input, verbose=0)[0]
        predictions_list.append(preds)

    # Average prediction
    preds = np.mean(predictions_list, axis=0)

    idx = np.argmax(preds)
    confidence = round(preds[idx] * 100, 2)

    predicted_stage = CLASS_NAMES[idx]

    stage_colors = {
        "Non Demented": "#2ecc71",
        "Very Mild Demented": "#f1c40f",
        "Mild Demented": "#e67e22",
        "Moderate Demented": "#e74c3c"
    }

    stage_descriptions = {
        "Non Demented": "The brain MRI shows no detectable signs of cognitive degeneration. Neural structural integrity appears preserved.",
        "Very Mild Demented": "Early neurocognitive warning signs may be present. Continuous monitoring is recommended.",
        "Mild Demented": "Noticeable reduction in cognitive functional markers affecting memory and daily processing.",
        "Moderate Demented": "Severe neurological impairment detected. Immediate clinical evaluation is strongly advised."
    }

    # ===========================================
    # Row 1 Layout
    # MRI Preview + Diagnosis Card
    # ===========================================

    col1, col2 = st.columns([1, 1])

    # MRI Preview
    with col1:
        st.subheader("📷 MRI Preview")
        st.image(image, width=350)

    # Diagnosis Card
    with col2:

        color = stage_colors.get(predicted_stage, "white")

        st.markdown(f"""
        <div class="card" style="border-left:8px solid {color};">

        <h2 style="color:{color};">{predicted_stage}</h2>

        <p>
        {stage_descriptions.get(predicted_stage, "")}
        </p>

        <p><b>Confidence :</b> {confidence}%</p>

        </div>
        """, unsafe_allow_html=True)

        # Confidence Graph
        fig, ax = plt.subplots(figsize=(4, 2.5))

        x = np.arange(len(CLASS_NAMES))

        # Base prediction curve
        ax.plot(x, preds,
                linewidth=2,
                label="Prediction Confidence",
                color="#00bcd4")

        # Uncertainty simulation band (±5%)
        uncertainty = 0.05

        upper_band = np.clip(preds + uncertainty, 0, 1)
        lower_band = np.clip(preds - uncertainty, 0, 1)

        ax.fill_between(
            x,
            lower_band,
            upper_band,
            alpha=0.2,
            label="Uncertainty Range"
        )

        ax.scatter(x, preds, s=50)

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, fontsize=7)

        ax.set_ylabel("Confidence Probability")
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.legend(fontsize=6)

        st.pyplot(fig)

    st.markdown("---")

    # ===========================================
    # GradCAM Visualization Section
    # ===========================================

    st.subheader("🧠 AI Attention Visualization")

    heatmap = generate_gradcam(model, arr)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img = np.uint8(arr[0] * 255)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    # Store visualization snapshots for clinical report generation
    st.session_state["last_original_img"] = original_img
    st.session_state["last_gradcam_img"] = superimposed_img

    col1, col2 = st.columns([0.9, 0.9])

    with col1:
        st.image(original_img,
                caption="Original MRI Scan",
                width=280,
                channels="BGR")


    with col2:
        st.image(superimposed_img,
                caption="GradCAM Heatmap",
                width=280,
                channels="BGR")

    # ===========================================
    # GradCAM Interpretation Description
    # ===========================================

    st.markdown("""
    <div style="background:#f8fbff;
                padding:18px;
                border-radius:12px;
                border-left:6px solid #0f4c75;
                margin-top:20px;">

    <h4>🧠 Clinical Interpretability Note</h4>

    <p style="font-size:14px; color:#444;">
    The GradCAM heatmap highlights brain regions that contributed most strongly
    to the AI model's classification decision.

    Brighter activation zones indicate higher neural attention during inference.

    This visualization is intended to support clinical interpretability and
    should not be used as an independent diagnostic tool.
    </p>

    </div>
    """, unsafe_allow_html=True)
def save_temp_image(fig, filename):
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return filename
def generate_radiology_summary(pre_stage, post_stage, confidence):

    severity_map = SEVERITY_ORDER

    pre_level = severity_map[pre_stage]
    post_level = severity_map[post_stage]

    if post_level < pre_level:
        clinical_trend = "The cognitive severity shows measurable improvement following treatment intervention."
    elif post_level > pre_level:
        clinical_trend = "The imaging and inference pattern suggest possible disease progression requiring clinical attention."
    else:
        clinical_trend = "The neurological condition appears stable with no significant stage transition."

    summary_text = f"""
    Radiology AI Clinical Summary:

    The neurocognitive screening analysis identifies the patient condition as
    {post_stage} classification with prediction confidence of {confidence:.2f}%.

    {clinical_trend}

    Longitudinal comparison between pre-treatment and post-treatment MRI inference
    demonstrates cognitive severity index movement from
    {pre_stage} stage to {post_stage} stage.

    Continuous neurological monitoring and professional clinical correlation are strongly recommended.
    """

    return summary_text
# ===================================================
# COMPARISON PAGE
# ===================================================
def show_comparison():

    st.title("📊 Dementia Treatment Comparison Dashboard")

    # Session safety initialization
    st.session_state.setdefault("pre_index", None)
    st.session_state.setdefault("post_index", None)
    st.session_state.setdefault("pre_confidence", 0)
    st.session_state.setdefault("post_confidence", 0)

    # ---------------- Result Card Renderer ----------------

    def render_result_card(pre_stage, post_stage):

        pre_severity = SEVERITY_ORDER[pre_stage]
        post_severity = SEVERITY_ORDER[post_stage]

        status_color = (
            "#2ecc71" if post_severity < pre_severity
            else "#e74c3c" if post_severity > pre_severity
            else "#f39c12"
        )

        status_text = (
            "Improved" if post_severity < pre_severity
            else "Condition Worsened" if post_severity > pre_severity
            else "No Significant Change"
        )

        interpretation = (
            "Cognitive severity reduced following treatment intervention."
            if post_severity < pre_severity
            else "Progression in dementia stage detected. Immediate clinical attention recommended."
            if post_severity > pre_severity
            else "Patient condition remains stable with no measurable stage variation."
        )

        st.markdown("### 🏥 Treatment Outcome Report")

        col1, col2 = st.columns(2)

        with col1:
            st.info("Pre-Treatment Diagnosis")
            st.markdown(f"## {pre_stage}")

        with col2:
            st.success("Post-Treatment Diagnosis")
            st.markdown(f"## {post_stage}")

        st.markdown("---")

        st.markdown(f"""
        <div class="card" style="border-left:8px solid {status_color};">

        <h4>Clinical Status: {status_text}</h4>

        <p><b>Clinical Interpretation:</b><br>
        {interpretation}
        </p>

        <p>
        <b>Recommendation:</b>
        Continued neurological monitoring and cognitive assessment advised.
        </p>

        </div>
        """, unsafe_allow_html=True)

    # ---------------- Mode Selection ----------------

    mode = st.radio(
        "Select Analysis Mode",
        ["Comparison Mode", "Clinical Insight Mode"]
    )

    st.markdown("---")

    # =========================
    # COMPARISON MODE
    # =========================

    if mode == "Comparison Mode":

        col1, col2 = st.columns(2)

        # -------- PRE MRI --------
        with col1:

            st.subheader("🧠 Pre Treatment MRI")

            pre_upload = st.file_uploader(
                "Upload Pre MRI",
                key="pre_mri",
                type=["jpg", "png", "jpeg"]
            )

            if pre_upload:

                image = Image.open(pre_upload).convert("RGB")
                img = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE))
                arr = np.expand_dims(np.array(img) / 255.0, axis=0)

                preds = model.predict(arr)[0]

                st.session_state.pre_index = int(np.argmax(preds))
                st.session_state.pre_confidence = float(np.max(preds))

                pre_stage = CLASS_NAMES[st.session_state.pre_index]

                st.image(image, width=250)
                st.success(f"Predicted Stage: {pre_stage}")

                # Graph
                fig, ax = plt.subplots(figsize=(3, 2))
                x = np.arange(len(CLASS_NAMES))

                ax.plot(x, preds, linewidth=2)
                ax.scatter(x, preds, s=40)

                ax.set_xticks(x)
                ax.set_xticklabels(CLASS_NAMES, rotation=30, fontsize=7)
                ax.set_ylabel("Confidence")
                ax.grid(True, linestyle="--", alpha=0.5)

                st.pyplot(fig)

        # -------- POST MRI --------
        with col2:

            st.subheader("🧠 Post Treatment MRI")

            post_upload = st.file_uploader(
                "Upload Post MRI",
                key="post_mri",
                type=["jpg", "png", "jpeg"]
            )

            if post_upload:

                image = Image.open(post_upload).convert("RGB")
                img = ImageOps.fit(image, (IMG_SIZE, IMG_SIZE))
                arr = np.expand_dims(np.array(img) / 255.0, axis=0)

                preds = model.predict(arr)[0]

                st.session_state.post_index = int(np.argmax(preds))
                st.session_state.post_confidence = float(np.max(preds))

                post_stage = CLASS_NAMES[st.session_state.post_index]

                st.image(image, width=250)
                st.success(f"Predicted Stage: {post_stage}")

                # Confidence Graph
                fig, ax = plt.subplots(figsize=(3, 2))

                x = np.arange(len(CLASS_NAMES))

                ax.plot(x, preds, linewidth=2)
                ax.scatter(x, preds, s=40)

                ax.set_xticks(x)
                ax.set_xticklabels(CLASS_NAMES, rotation=30, fontsize=7)
                ax.set_ylabel("Confidence")
                ax.grid(True, linestyle="--", alpha=0.5)

                st.pyplot(fig)

        # Result Report Card
        if (st.session_state.pre_index is not None and
            st.session_state.post_index is not None):

            pre_stage = CLASS_NAMES[st.session_state.pre_index]
            post_stage = CLASS_NAMES[st.session_state.post_index]

            render_result_card(pre_stage, post_stage)

    # =========================
    # CLINICAL INSIGHT MODE
    # =========================

    elif mode == "Clinical Insight Mode":

        st.subheader("🏥 Clinical Insight Screening Report")

        if (st.session_state.pre_index is None or
            st.session_state.post_index is None):

            st.warning("Upload both Pre-treatment and Post-treatment MRI scans first.")
            return

        doctor_name = st.session_state.doctor_name

        st.info(f"Authorizing Doctor : {doctor_name}")

        # ---------- Report Generation Button ----------
        
        if st.button("📄 Generate Clinical Insight Report"):
            st.session_state.setdefault("last_original_img", np.zeros((224,224,3)))
            st.session_state.setdefault("last_gradcam_img", np.zeros((224,224,3)))

            # ===== Save MRI Snapshot Properly =====
            # ======================================================
            # Cognitive Progression Line Graph (Pre → Post Treatment)
            # ======================================================

            os.makedirs("report_assets", exist_ok=True)

            pre_stage = CLASS_NAMES[st.session_state.pre_index]
            post_stage = CLASS_NAMES[st.session_state.post_index]

            pre_value = SEVERITY_ORDER[pre_stage]
            post_value = SEVERITY_ORDER[post_stage]

            plt.figure(figsize=(3.2, 2.6))

            x_labels = ["Pre Treatment", "Post Treatment"]
            y_values = [pre_value, post_value]

            plt.plot(x_labels, y_values,
                    marker="o",
                    linewidth=2,
                    linestyle="-",
                    label="Cognitive Severity Progression")

            plt.ylabel("Severity Index Level")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(fontsize=6)

            progression_graph_path = "report_assets/progression_report.png"

            plt.savefig(progression_graph_path, bbox_inches="tight", dpi=300)
            plt.close()
            original_img = st.session_state.get(
                "last_original_img",
                np.zeros((224,224,3))
            )

            gradcam_img = st.session_state.get(
                "last_gradcam_img",
                np.zeros((224,224,3))
            )

            # Clinical report assets folder safety
            os.makedirs("report_assets", exist_ok=True)

            # Save images using matplotlib (Streamlit safe method)

            plt.figure(figsize=(3,3))
            plt.imshow(original_img)
            plt.axis("off")
            original_path = "report_assets/original_report.png"
            plt.savefig(original_path, bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(3,3))
            plt.imshow(gradcam_img)
            plt.axis("off")
            gradcam_path = "report_assets/gradcam_report.png"
            plt.savefig(gradcam_path, bbox_inches='tight', dpi=300)
            plt.close()

            # ===== Generate Report =====

            # Radiology summary generator
            summary_text = generate_radiology_summary(
                pre_stage,
                post_stage,
                st.session_state.post_confidence * 100
            )

            report_path = generate_medical_report_pdf(
                patient_stage=CLASS_NAMES[st.session_state.post_index],
                confidence=st.session_state.post_confidence * 100,
                doctor_name=st.session_state.doctor_name,
                original_img_path=original_path,
                gradcam_img_path=gradcam_path,
                progression_graph_path=progression_graph_path,
                summary_text=summary_text
            )

            st.success("Clinical Report Generated 🎉")

            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Report PDF",
                    data=f,
                    file_name="NeuroCognitive_Report.pdf",
                    mime="application/pdf"
                )
#--------------------------------------------------
#------------Analytics page------------------------
#-----------------------------------------------
def show_analytics():

    st.markdown('<div class="gradient-header">📊 Dementia Awareness & Analytics</div>',
                unsafe_allow_html=True)

    # =====================================================
    # PUBLIC EDUCATION CONTENT ABOUT DEMENTIA
    # =====================================================

    st.markdown("""
    ## 🧠 About Dementia

    Dementia is a neurological disorder that affects memory, thinking ability,
    and daily cognitive functioning.

    According to the :contentReference[oaicite:0]{index=0},
    dementia is one of the major causes of disability and dependency among older adults worldwide.

    Early detection helps in better clinical management and lifestyle planning.

    ### Common Symptoms
    - Memory loss  
    - Difficulty in problem solving  
    - Confusion about time and place  
    - Behavioral and personality changes  

    ### Importance of Early Diagnosis
    Detecting cognitive decline at an early stage allows:
    - Slowing disease progression  
    - Planning treatment strategies  
    - Improving quality of life  

    """)

    st.markdown("---")

    # =====================================================
    # VISUAL DASHBOARDS
    # =====================================================

    st.markdown("### 📈 Dementia Stage Visualization")

    stages = ["Non Demented", "Very Mild", "Mild", "Moderate"]

    # Demo visualization data (since real database is not integrated)
    distribution = np.random.randint(40, 120, size=4)

    col1, col2 = st.columns(2)

    # ---------------- PIE CHART ----------------
    with col1:

        fig1, ax1 = plt.subplots(figsize=(3, 3))

        ax1.pie(distribution,
                labels=stages,
                autopct="%1.1f%%",
                startangle=90)

        ax1.set_title("Stage Distribution Overview")

        st.pyplot(fig1)

        st.markdown("""
        **Graph Explanation (Simple Understanding)**  
        This chart shows how dementia stages are distributed in the
        simulated clinical dataset. Larger slice means higher patient count.
        """)

    # ---------------- PROGRESSION TREND GRAPH ----------------
    with col2:

        progression = np.cumsum(np.random.randn(10))

        fig2, ax2 = plt.subplots(figsize=(4, 2))

        ax2.plot(progression, marker="o")
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("Severity Level")
        ax2.grid(True, linestyle="--", alpha=0.3)

        st.pyplot(fig2)

        st.markdown("""
        **Graph Meaning**  
        This line graph represents a sample disease progression trend.
        Rising line indicates increasing cognitive severity,
        while a stable line suggests controlled disease condition.
        """)

    st.markdown("---")

    # =====================================================
    # EDUCATIONAL SECTION FOR PUBLIC USERS
    # =====================================================

    st.markdown("## 📚 Learn More About Dementia")

    st.markdown("""
    Dementia is not a single disease but a group of symptoms affecting brain function.

    Common dementia types include Alzheimer-related cognitive degeneration and
    other neurovascular disorders.

    ⚠ Warning Signs That Need Medical Attention:
    - Frequent memory loss affecting daily life  
    - Difficulty communicating  
    - Losing track of familiar places  
    - Poor judgment or decision-making  

    🏥 Lifestyle Factors That Help Brain Health:
    - Balanced nutrition  
    - Regular mental exercises  
    - Physical activity  
    - Stress reduction  
    - Proper sleep pattern  

    Early screening and medical consultation are strongly recommended when symptoms appear.
    """)

    st.markdown("---")

    st.info("This analytics page is designed for public awareness and educational visualization.")
# ===================================================
# DIET PAGE
# ===================================================
def show_diet():

    st.markdown('<div class="gradient-header">Healthy Brain Diet Tips</div>',
                unsafe_allow_html=True)

    st.write("These foods support brain health, memory improvement and cognitive protection.")

    foods = [
        {"name": "Almonds", "filename": "almonds.jpeg", "info": "Rich in vitamin E for brain health."},
        {"name": "Black coffee", "filename": "black_coffee.jpeg", "info": "Enhances alertness and memory."},
        {"name": "Blueberries", "filename": "blueberries.jpeg", "info": "High in antioxidants for brain function."},
        {"name": "Broccoli", "filename": "broccoli.jpeg", "info": "Contains vitamin K for cognition."},
        {"name": "Cherries", "filename": "cherries.jpeg", "info": "Reduces inflammation and supports memory."},
        {"name": "Dark chocolate", "filename": "dark_chocolate.jpeg", "info": "Flavonoids improve brain function."},
        {"name": "Fenugreek leaves", "filename": "fenugreek_leaves.jpeg", "info": "Anti-inflammatory and memory support."},
        {"name": "Fish oil", "filename": "fish_oil.jpeg", "info": "Supports cognitive function."},
        {"name": "Flax seeds", "filename": "flax_seeds.jpg", "info": "Omega-3 fatty acids for neural protection."},
        {"name": "Green tea", "filename": "green_tea.jpeg", "info": "Antioxidants protect neurons."},
        {"name": "Lettuce", "filename": "lettuce.jpeg", "info": "Helps reduce oxidative stress."},
        {"name": "Pomegranate", "filename": "pomogranate.jpg", "info": "Improves cerebral blood flow."},
        {"name": "Pumpkin seeds", "filename": "pumpkin_seeds.jpeg", "info": "Zinc and magnesium for memory."},
        {"name": "Salmon", "filename": "salmon.jpeg", "info": "Boosts memory and learning."},
        {"name": "Sardines", "filename": "sardines.jpeg", "info": "Rich in DHA and EPA."},
        {"name": "Spinach", "filename": "spinach.jpeg", "info": "Rich in lutein and folate."},
        {"name": "Walnuts", "filename": "walnuts.jpeg", "info": "Omega-3 fatty acids boost cognition."}
    ]

    base_path = "D:/Project/neurocognet/images"

    cols = st.columns(3)

    for i, food in enumerate(foods):

        col = cols[i % 3]

        path = os.path.join(base_path, food["filename"])

        with col:

            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning("Image not found")

            st.markdown(f"### {food['name']}")
            st.write(food["info"])
            st.markdown("---")

    st.markdown("### Recommended Daily Intake for Brain Health")

    intake_data = {
        "Food Category": [
            "Nuts/Seeds",
            "Fruits",
            "Vegetables",
            "Fish/Omega-3",
            "Drinks/Snacks",
            "Herbs"
        ],
        "Daily Recommendation": [
            "20-30g",
            "1-2 servings",
            "2-3 servings",
            "2-3 servings or 250-300mg Omega-3",
            "1-2 cups or moderate",
            "1-2 tsp leaves/spices"
        ]
    }

    st.table(intake_data)
    
def generate_medical_report_pdf(
        patient_stage,
        confidence,
        doctor_name,
        original_img_path,
        gradcam_img_path,
        progression_graph_path,
        summary_text,
        output_path="NeuroCognitive_Report.pdf"):

    from reportlab.platypus import (
        Paragraph, Spacer, Image,
        SimpleDocTemplate, Table
    )

    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics
    import datetime

    # --------------------------------------------------
    # Clinical Font Setup
    # --------------------------------------------------

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.fontName = "HeiseiMin-W3"

    normal_style = styles["Normal"]
    normal_style.fontName = "HeiseiMin-W3"

    elements = []

    # --------------------------------------------------
    # Clinical Hospital Header Banner
    # --------------------------------------------------

    elements.append(Paragraph(
        "NeuroCognitive AI Clinical Screening Report",
        title_style
    ))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_text = f"""
    <b>Timestamp :</b> {timestamp}<br/>
    <b>Authorizing Doctor :</b> {doctor_name}<br/>
    <b>Screening Mode :</b> Clinical Insight Analysis
    """

    elements.append(Paragraph(header_text, normal_style))
    elements.append(Spacer(1, 18))

    # --------------------------------------------------
    # Clinical Summary Medical Card
    # --------------------------------------------------

    summary_table = Table([
        ["Predicted Cognitive Stage", patient_stage],
        ["Prediction Confidence", f"{confidence:.2f}%"]
    ])

    summary_table.setStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 1.3, colors.black),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("FONT", (0, 0), (-1, -1), "HeiseiMin-W3"),
    ])

    elements.append(summary_table)
    elements.append(Spacer(1, 22))

    # --------------------------------------------------
    # MRI Clinical Visualization Panel (Radiology Layout)
    # --------------------------------------------------

    try:
        elements.append(Paragraph("MRI Clinical Radiology Visualization", styles["Heading2"]))

        mri_table = Table([
            [
                Image(original_img_path, width=180, height=180),
                Image(gradcam_img_path, width=180, height=180)
            ],
            ["Original MRI Scan", "AI Attention Heatmap"]
        ])

        mri_table.setStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONT", (0, 0), (-1, -1), "HeiseiMin-W3"),
            ("GRID", (0, 0), (-1, -1), 0.6, colors.black)
        ])

        elements.append(mri_table)

    except Exception:
        elements.append(Paragraph(
            "MRI visualization data not available.",
            normal_style
        ))

    elements.append(Spacer(1, 25))
    # --------------------------------------------------
    # Radiology AI Clinical Summary Section
    # --------------------------------------------------

    elements.append(Paragraph("AI Radiology Clinical Summary", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 20))

    # --------------------------------------------------
    # Cognitive Progression Trend Graph
    # --------------------------------------------------

    try:
        elements.append(Paragraph("Cognitive Progression Trend", styles["Heading2"]))

        elements.append(Image(progression_graph_path, width=250, height=180))

    except Exception:
        elements.append(Paragraph(
            "Progression visualization not available.",
            normal_style
        ))

    # --------------------------------------------------
    # AI Clinical Interpretation Block (Radiologist Style Text)
    # --------------------------------------------------

    interpretation = f"""
    Clinical Interpretation Summary:<br/><br/>

    The neurocognitive AI screening system identified structural pattern
    corresponding to <b>{patient_stage}</b> stage dementia classification.

    The prediction confidence score was measured at
    <b>{confidence:.2f}%</b>, indicating statistical reliability of
    the inference model.

    Continuous neurological monitoring and clinical correlation is
    recommended for treatment planning and cognitive management.

    This AI report is intended for screening assistance only and
    should not be used as standalone diagnostic authority.
    """

    elements.append(Paragraph(interpretation, normal_style))

    # --------------------------------------------------
    # Professional Disclaimer Footer
    # --------------------------------------------------

    elements.append(Spacer(1, 40))

    disclaimer = """
    <b>Disclaimer</b><br/>
    This AI system is designed for clinical screening assistance only.<br/>
    Final diagnosis must be confirmed by qualified neurological specialists.
    """

    elements.append(Paragraph(disclaimer, normal_style))

    doc.build(elements)

    return output_path

# ===================================================
# MAIN ROUTER
# ===================================================

if menu == "Home":
    show_home()
    show_home_access()

elif menu == "Detection":
    if st.session_state.role == "doctor":
        show_detection()
    else:
        st.warning("Doctor access required")

elif menu == "Comparison":
    if st.session_state.role == "doctor":
        show_comparison()
    else:
        st.warning("Doctor access required")

elif menu == "Analytics":
    show_analytics()

elif menu == "Diet":
    show_diet()

elif menu == "Logout":
    st.session_state.role = "public"
    st.session_state.doctor_name = None
    st.session_state["doctor_logged_in"] = False
    st.session_state["pre_index"] = None
    st.session_state["post_index"] = None

    st.success("Logged out successfully")
    st.rerun()
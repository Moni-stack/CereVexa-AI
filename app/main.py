import streamlit as st
from assistant.assistant import render_assistant
from PIL import Image
import sys
import os


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MODELS
# ============================================================

from classification.classifier import (
    BrainTumorClassifier,
    TumorDetector
)

from segmentation.segmenter import BrainTumorSegmenter

from explainability.gradcam import BrainTumorGradCAM

from utils.image_utils import create_segmentation_overlay


# ============================================================
# MODEL PATHS
# ============================================================

CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "resnet18_brain_tumor_best.pth"
)

DETECTOR_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "tumor_detector_best.pth"
)

UNET_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "unet_brain_tumor_best.pth"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CereVexa AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS  (UI THEME ONLY)
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0b2545;
        --blue: #13538a;
        --teal: #0e9aa7;
        --teal-soft: #e4f6f7;
        --bg: #f3f7fb;
        --card: #ffffff;
        --border: #e2e9f2;
        --text: #1d2b3a;
        --muted: #6b7a8c;
        --danger: #c0392b;
        --danger-soft: #fdecea;
        --ok: #1e8e5a;
        --ok-soft: #e6f6ee;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    .stApp {
        background: linear-gradient(180deg, #eef4fa 0%, #f7fafd 40%, #f3f7fb 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2545 0%, #0f3460 100%);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #dbe7f5 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
    }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .side-logo {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0e9aa7, #2bc4d1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 6px 16px rgba(14,154,167,0.35);
    }

    .side-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: 0.2px;
        line-height: 1.1;
    }

    .side-sub {
        font-size: 0.75rem;
        color: #9fb6d3 !important;
    }

    .side-heading {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #7fd4db !important;
        margin: 6px 0 10px 0;
    }

    .side-step {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
        align-items: flex-start;
    }

    .side-num {
        min-width: 26px;
        height: 26px;
        border-radius: 50%;
        background: rgba(43,196,209,0.18);
        border: 1px solid rgba(43,196,209,0.5);
        color: #7fe3ec !important;
        font-size: 0.78rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .side-step-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #ffffff !important;
    }

    .side-step-desc {
        font-size: 0.78rem;
        color: #9fb6d3 !important;
        line-height: 1.35;
    }

    .side-model {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 8px;
    }

    .side-model b {
        color: #ffffff !important;
        font-size: 0.84rem;
    }

    .side-model span {
        display: block;
        font-size: 0.75rem;
        color: #9fb6d3 !important;
    }

    .side-notice {
        background: rgba(255, 193, 7, 0.10);
        border: 1px solid rgba(255, 193, 7, 0.35);
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.75rem;
        line-height: 1.45;
        color: #f6e3a1 !important;
    }


    /* ---------- Hero ---------- */

    .hero {
        background: linear-gradient(120deg, #0b2545 0%, #13538a 55%, #0e9aa7 100%);
        border-radius: 22px;
        padding: 34px 40px;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 14px 40px rgba(11,37,69,0.22);
        margin-bottom: 22px;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.18), rgba(255,255,255,0) 70%);
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero h1 {
        color: white !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        padding: 0;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: #d3e6f7;
        font-size: 1.02rem;
        margin: 0;
        max-width: 640px;
        line-height: 1.55;
    }

    .hero-status {
        margin-top: 18px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(255,255,255,0.14);
        padding: 6px 14px;
        border-radius: 999px;
    }

    .dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 0 4px rgba(74,222,128,0.25);
    }


    /* ---------- Feature cards ---------- */

    .feat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 26px;
    }

    .feat {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(20,50,90,0.05);
    }

    .feat-ico {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: var(--teal-soft);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-bottom: 10px;
    }

    .feat-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--navy);
    }

    .feat-desc {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 3px;
        line-height: 1.4;
    }


    /* ---------- Section headers ---------- */

    .sec {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 8px 0 4px 0;
    }

    .sec-badge {
        min-width: 38px;
        height: 38px;
        border-radius: 11px;
        background: linear-gradient(135deg, #13538a, #0e9aa7);
        color: white;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 6px 14px rgba(19,83,138,0.25);
    }

    .sec-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.15;
    }

    .sec-sub {
        font-size: 0.86rem;
        color: var(--muted);
        margin: 0 0 14px 52px;
    }


    /* ---------- Result banners ---------- */

    .banner {
        border-radius: 16px;
        padding: 18px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 6px 0 16px 0;
        border: 1px solid;
    }

    .banner-ico {
        font-size: 28px;
    }

    .banner-title {
        font-weight: 800;
        font-size: 1.1rem;
    }

    .banner-sub {
        font-size: 0.85rem;
        opacity: 0.85;
    }

    .banner.ok {
        background: var(--ok-soft);
        border-color: #bfe6d0;
        color: #14653f;
    }

    .banner.alert {
        background: var(--danger-soft);
        border-color: #f4c7c1;
        color: #912d22;
    }

    .note {
        background: #eef5fb;
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #2c4763;
        margin-top: 12px;
    }

    .caption-box {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 8px;
        line-height: 1.45;
    }


    /* ---------- Pipeline tracker ---------- */

    .pipe {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 16px rgba(20,50,90,0.05);
    }

    .pipe-title {
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 4px;
    }

    .pipe-sub {
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 14px;
    }

    .pipe-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 0;
        border-top: 1px dashed var(--border);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .pipe-row:first-of-type {
        border-top: none;
    }

    .pipe-ico {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: var(--teal-soft);
        display: flex;
        align-items: center;
        justify-content: center;
    }


    /* ---------- Streamlit widgets ---------- */

    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #b9d3e8;
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 4px 18px rgba(30, 50, 80, 0.05);
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--teal);
        background-color: #fbfeff;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfe 100%);
        border: 1px solid var(--border);
        border-top: 4px solid var(--teal);
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(30, 50, 80, 0.05);
    }

    [data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: var(--navy);
        font-size: 1.7rem;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(11,37,69,0.14);
        border: 1px solid var(--border);
    }

    [data-testid="stImageCaption"] {
        font-size: 0.8rem;
        color: var(--muted);
        font-weight: 500;
    }

    .stButton button {
        border-radius: 10px;
    }

    hr {
        border-color: #e1e8f0;
        margin: 1.6rem 0;
    }

    .foot {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.7;
        padding: 10px 0 0 0;
    }

    .foot b {
        color: var(--navy);
    }

    @media (max-width: 900px) {
        .feat-grid { grid-template-columns: repeat(2, 1fr); }
        .hero { padding: 24px; }
        .hero h1 { font-size: 1.8rem; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UI HELPERS (presentation only)
# ============================================================

def section_header(number, title, subtitle):

    st.markdown(
        f"""<div class="sec"><div class="sec-badge">{number}</div><div class="sec-title">{title}</div></div><div class="sec-sub">{subtitle}</div>""",
        unsafe_allow_html=True
    )


def banner(kind, icon, title, subtitle):

    st.markdown(
        f"""<div class="banner {kind}"><div class="banner-ico">{icon}</div><div><div class="banner-title">{title}</div><div class="banner-sub">{subtitle}</div></div></div>""",
        unsafe_allow_html=True
    )


def caption(text):

    st.markdown(
        f"""<div class="caption-box">{text}</div>""",
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">
            <div class="side-logo">🧠</div>
            <div>
                <div class="side-title">CereVexa AI</div>
                <div class="side-sub">Explainable Brain MRI Analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="side-heading">How it works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-step"><div class="side-num">1</div><div>
            <div class="side-step-title">Tumor Detection</div>
            <div class="side-step-desc">Determines whether a tumor is present.</div>
        </div></div>
        <div class="side-step"><div class="side-num">2</div><div>
            <div class="side-step-title">Classification</div>
            <div class="side-step-desc">Identifies the predicted tumor type.</div>
        </div></div>
        <div class="side-step"><div class="side-num">3</div><div>
            <div class="side-step-title">Localization</div>
            <div class="side-step-desc">U-Net estimates the tumor region.</div>
        </div></div>
        <div class="side-step"><div class="side-num">4</div><div>
            <div class="side-step-title">Explainability</div>
            <div class="side-step-desc">Grad-CAM highlights regions influencing classification.</div>
        </div></div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="side-heading">AI Models</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-model"><b>Tumor Detector</b><span>ResNet18 — Binary Classification</span></div>
        <div class="side-model"><b>Tumor Classifier</b><span>ResNet18 — 3-Class Classification</span></div>
        <div class="side-model"><b>Localization</b><span>U-Net</span></div>
        <div class="side-model"><b>Explainability</b><span>Grad-CAM</span></div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div class="side-notice">
            <b>⚠️ Research Notice</b><br>
            CereVexa AI is an educational and research
            prototype. Results are not intended to be
            used as a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">Clinical Decision Support Prototype</div>
        <h1>🧠 CereVexa AI</h1>
        <p>
            Explainable Brain MRI Analysis — an AI-assisted pipeline for
            tumor detection, classification, localization and explainability.
        </p>
        <div class="hero-status"><span class="dot"></span> AI Analysis System Ready</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="feat-grid">
        <div class="feat"><div class="feat-ico">🔍</div>
            <div class="feat-title">Detection</div>
            <div class="feat-desc">Screens the scan for tumor presence.</div></div>
        <div class="feat"><div class="feat-ico">🧬</div>
            <div class="feat-title">Classification</div>
            <div class="feat-desc">Predicts the tumor category.</div></div>
        <div class="feat"><div class="feat-ico">🎯</div>
            <div class="feat-title">Localization</div>
            <div class="feat-desc">Pixel-level estimate of tumor region.</div></div>
        <div class="feat"><div class="feat-ico">🔥</div>
            <div class="feat-title">Explainability</div>
            <div class="feat-desc">Visual reasoning behind predictions.</div></div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

section_header(
    "↑",
    "Analyze a Brain MRI",
    "Upload a brain MRI scan to detect tumor presence, classify the "
    "tumor type, estimate its location, and understand the model's prediction."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    detector = TumorDetector(
        DETECTOR_PATH
    )

    classifier = BrainTumorClassifier(
        CLASSIFIER_PATH
    )

    segmenter = BrainTumorSegmenter(
        UNET_PATH
    )

    gradcam = BrainTumorGradCAM(
        classifier.model
    )

    return (
        detector,
        classifier,
        segmenter,
        gradcam
    )


detector, classifier, segmenter, gradcam = load_models()


# ============================================================
# UPLOAD MRI
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    help="Upload a brain MRI image."
)

st.caption(
    "Supported formats: PNG, JPG, JPEG"
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    st.divider()

    section_header(
        "•",
        "MRI Scan",
        "The uploaded scan that will be analyzed."
    )

    image_col, info_col = st.columns(
        [1.2, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded MRI",
            width=500
        )

    with info_col:

        st.markdown(
            """
            <div class="pipe">
                <div class="pipe-title">Analysis Pipeline</div>
                <div class="pipe-sub">The MRI will pass through the CereVexa AI analysis pipeline.</div>
                <div class="pipe-row"><div class="pipe-ico">🔍</div>Tumor Detection</div>
                <div class="pipe-row"><div class="pipe-ico">🧬</div>Tumor Classification</div>
                <div class="pipe-row"><div class="pipe-ico">🎯</div>Tumor Localization</div>
                <div class="pipe-row"><div class="pipe-ico">🔥</div>Grad-CAM Explainability</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # STEP 1 — DETECTION
    # ========================================================

    st.divider()

    section_header(
        "1",
        "Tumor Detection",
        "Binary screening for tumor presence."
    )

    with st.spinner("Analyzing tumor presence..."):

        detection = detector.predict(
            image
        )


    # ========================================================
    # NO TUMOR
    # ========================================================

    if detection["class"] == "No Tumor":

        banner(
            "ok",
            "✅",
            "No tumor detected",
            "Result from the tumor-presence model."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Detection",
                "No Tumor"
            )

        with col2:

            st.metric(
                "Model Confidence",
                f"{detection['confidence'] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Pipeline Status",
                "Complete"
            )

        st.markdown(
            """
            <div class="note">
                The tumor detector predicted No Tumor, so
                tumor classification, U-Net localization
                and Grad-CAM analysis were skipped.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # TUMOR DETECTED
    # ========================================================

    else:

        banner(
            "alert",
            "⚠️",
            "Tumor detected",
            "Result from the tumor-presence model. Continuing analysis."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Detection",
                "Tumor"
            )

        with col2:

            st.metric(
                "Tumor Probability",
                f"{detection['probability'] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Pipeline Status",
                "Analyzing"
            )


        # ====================================================
        # STEP 2 — CLASSIFICATION
        # ====================================================

        st.divider()

        section_header(
            "2",
            "Tumor Classification",
            "The ResNet18 classifier predicts the tumor category."
        )

        with st.spinner("Classifying tumor type..."):

            result = classifier.predict(
                image
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Tumor Type",
                result["class"]
            )

        with col2:

            st.metric(
                "Classification Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )


        # ====================================================
        # STEP 3 — U-NET
        # ====================================================

        st.divider()

        section_header(
            "3",
            "Tumor Localization",
            "U-Net generates a pixel-level mask estimating the tumor region."
        )

        with st.spinner("Generating tumor localization..."):

            mask = segmenter.predict(
                image
            )

        overlay = create_segmentation_overlay(
            image,
            mask
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original MRI",
                width=500
            )

        with col2:

            st.image(
                overlay,
                caption="U-Net Tumor Localization",
                width=500
            )

        tumor_area = mask.mean() * 100

        st.metric(
            "Estimated Segmented Area",
            f"{tumor_area:.2f}%"
        )

        caption(
            "The segmented area represents the proportion "
            "of the processed image covered by the predicted "
            "tumor mask. It is not a clinical measurement."
        )


        # ====================================================
        # STEP 4 — GRAD-CAM
        # ====================================================

        st.divider()

        section_header(
            "4",
            "Classification Explainability",
            "Grad-CAM highlights regions that influenced the classifier's prediction."
        )

        class_names = [
            "Meningioma",
            "Glioma",
            "Pituitary"
        ]

        class_index = class_names.index(
            result["class"]
        )

        with st.spinner("Generating Grad-CAM explanation..."):

            cam_image = gradcam.generate(
                image,
                class_index
            )

        st.image(
            cam_image,
            caption="Grad-CAM — Classification Explanation",
            width=650
        )

        caption(
            "Grad-CAM highlights regions influencing the "
            "classification prediction. It is not a "
            "pixel-level tumor segmentation."
        )


        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        st.divider()

        section_header(
            "✓",
            "Analysis Summary",
            "Overview of the AI findings for this scan."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Tumor Detection",
                "Tumor"
            )

        with col2:

            st.metric(
                "Tumor Type",
                result["class"]
            )

        with col3:

            st.metric(
                "Classification Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()
render_assistant()

st.markdown(
    """
    <div class="foot">
        <b>🧠 CereVexa AI</b> • Explainable Brain MRI Analysis<br>
        Educational &amp; Research Prototype • Not intended for medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)


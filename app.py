import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre

# CONFIGURATION

# Chemin vers le dossier contenant tes modèles .h5
MODELS_DIR = "." 

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6

CLASS_NAMES = [
    'crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole', 
    'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line'
]

# PAGE CONFIG

st.set_page_config(
    page_title="BENCHMARK DÉFAUTS METALLIQUES",
    page_icon="🔍",
    layout="wide"
)

# CSS GLOBAL
st.markdown("""
<style>
.card {
    padding: 1.2rem;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e1e5eb;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.result-ok {
    background-color: #e8f5e9;
    border-left: 6px solid #2e7d32;
}

.result-warn {
    background-color: #fff3e0;
    border-left: 6px solid #ef6c00;
}

.result-danger {
    background-color: #ffebee;
    border-left: 6px solid #c62828;
}

.model-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 0.5rem;
}

.big-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODELS

@st.cache_resource
def load_models():
    """Charge les 3 modèles et gère les erreurs de fichier manquant."""
    models = {}
    model_files = {
        "MobileNetV2": "mobilenetv2_gc10_finetuned.h5",
        "EfficientNetB0": "efficientnetb0_gc10_finetuned.h5",
        "ResNet50": "resnet50_gc10_finetuned.h5"
    }

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            st.error(f"❌ Fichier introuvable : {filename}")
            st.stop()
        try:
            models[name] = tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Erreur chargement {name}: {e}")
            st.stop()
    return models

models = load_models()

# FONCTIONS UTILITAIRES

def get_preprocessing_function(model_name):
    if model_name == "MobileNetV2": return mn_pre
    elif model_name == "EfficientNetB0": return eff_pre
    elif model_name == "ResNet50": return res_pre
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_index=1, pred_index=None):
    """Génération robuste de Heatmap pour modèles imbriqués (Backbone + Head)"""
    try:
        backbone = model.layers[last_conv_layer_index]
        classifier_layers = model.layers[last_conv_layer_index + 1:]

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs = backbone(inputs)
            tape.watch(conv_outputs)
            
            x = conv_outputs
            for layer in classifier_layers:
                x = layer(x)
            preds = x
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        return None

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    if heatmap is None: return np.array(original_img)
    img = np.array(original_img)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    return np.uint8(heatmap * alpha + img * (1 - alpha))

with st.sidebar:
    st.header("⚙️ Paramètres Benchmark")

    st.markdown("### 📌 Modèles comparés")
    st.write(f"- MobileNetV2")
    st.write(f"- EfficientNetB0")
    st.write(f"- ResNet50")
    
    st.markdown("### 🎯 Configuration")
    st.write(f"- Taille entrée : **{IMG_SIZE[0]}×{IMG_SIZE[1]}**")
    
    # Sélecteur de seuil interactif
    current_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.info("Comparaison multi-modèles avec explicabilité Grad-CAM.")

# HEADER

st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0px;'>
        🔬 COMPARAISON DES DÉFAUTS METALLIQUES <br> ENTRE 03 MODELES DE TRANSFERT LEARNING
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; margin-top: 10px;'>
        Comparaison de performances entre <strong>MobileNetV2</strong>, <strong>EfficientNetB0</strong> et <strong>ResNet50</strong>.
    </p>
""", unsafe_allow_html=True)

# UPLOAD IMAGE

uploaded_file = st.file_uploader(
    "📤 Charger une image à analyser",
    type=["jpg", "jpeg", "png"],
    help="Image couleur ou niveaux de gris"
)

image_loaded = uploaded_file is not None

# TABS (Analyse / Détails / Export)

tab_analysis, tab_details, tab_export = st.tabs(
    ["🔍 ANALYSE & GRAD-CAM", "📊 DÉTAILS COMPARATIFS", "⬇️ EXPORT DONNÉES"]
)

if image_loaded:
    original_image = Image.open(uploaded_file).convert('RGB')
    img_resized = original_image.resize(IMG_SIZE)
    img_array_base = tf.keras.preprocessing.image.img_to_array(img_resized)

# TAB 1: ANALYSE & GRAD-CAM

with tab_analysis:
    st.markdown("<div class='section-title'>🔍 Comparaison Visuelle</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Veuillez charger une image.</div>", unsafe_allow_html=True)
    else:
        st.image(original_image, caption="Image originale", width=300)
        st.markdown("---")
        
        cols = st.columns(3)
        
        # Dictionnaire pour stocker les résultats pour l'export et les graphes
        results_store = {}

        for idx, (model_name, model) in enumerate(models.items()):
            with cols[idx]:
                # 1. Prédiction
                preprocess = get_preprocessing_function(model_name)
                img_input = np.expand_dims(img_array_base.copy(), axis=0)
                img_preprocessed = preprocess(img_input)
                
                preds = model.predict(img_preprocessed, verbose=0)
                confidence = np.max(preds[0])
                class_idx = np.argmax(preds[0])
                class_name = CLASS_NAMES[class_idx]
                
                # Stockage
                results_store[model_name] = {
                    "class": class_name,
                    "confidence": float(confidence),
                    "probs": preds[0]
                }

                # 2. Style de la carte résultat
                if confidence >= current_threshold:
                    css = "result-ok"
                    icon = "✅"
                elif confidence >= 0.4:
                    css = "result-warn"
                    icon = "⚠️"
                else:
                    css = "result-danger"
                    icon = "❓"

                # 3. Affichage Carte
                st.markdown(f"""
                <div class="card {css}">
                    <div class="model-header">{model_name}</div>
                    <div class="big-title">{icon} {class_name}</div>
                    <p>Confiance : <strong>{confidence:.2%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                # 4. Grad-CAM
                heatmap = make_gradcam_heatmap(img_preprocessed, model)
                if heatmap is not None:
                    overlay = overlay_heatmap(heatmap, img_resized)
                    st.image(overlay, caption=f"Focus {model_name}", use_container_width=True)
                else:
                    st.warning("Grad-CAM indisponible")

# TAB 2: DÉTAILS COMPARATIFS
with tab_details:
    st.markdown("<div class='section-title'>📊 Comparaison des Probabilités</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Analyse requise.</div>", unsafe_allow_html=True)
    else:
        # Graphique comparatif
        st.subheader("Distribution des probabilités par modèle")
        
        # On crée 3 sous-graphiques
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        
        for idx, (model_name, data) in enumerate(results_store.items()):
            ax = axes[idx]
            probs = data["probs"]
            
            # Bar chart
            bars = ax.barh(CLASS_NAMES, probs, color='#1f77b4')
        
            max_idx = np.argmax(probs)
            bars[max_idx].set_color('#2ca02c') # Vert pour la classe prédite
            
            ax.set_title(f"{model_name}\n({data['class']})")
            ax.set_xlim(0, 1)
            ax.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        
        # Tableau récapitulatif
        st.subheader("Tableau de synthèse")
        summary_data = []
        for name, data in results_store.items():
            row = {"Modèle": name, "Prédiction": data["class"], "Confiance": f"{data['confidence']:.2%}"}
            summary_data.append(row)
        st.table(summary_data)

# TAB 3: EXPORT
with tab_export:
    st.markdown("<div class='section-title'>⬇️ Export des Résultats</div>", unsafe_allow_html=True)

    if not image_loaded:
        st.markdown("<div class='card disabled-step'>Analyse requise avant export.</div>", unsafe_allow_html=True)
    else:
        # Préparation du JSON propre
        export_data = {
            "filename": uploaded_file.name,
            "models_comparison": {}
        }
        
        for name, data in results_store.items():
            export_data["models_comparison"][name] = {
                "predicted_class": data["class"],
                "confidence": data["confidence"],
                "all_probabilities": {
                    k: float(v) for k, v in zip(CLASS_NAMES, data["probs"])
                }
            }

        st.json(export_data, expanded=False)

        st.download_button(
            label="📥 Télécharger le rapport JSON complet",
            data=json.dumps(export_data, indent=2),
            file_name="benchmark_result.json",
            mime="application/json"
        )
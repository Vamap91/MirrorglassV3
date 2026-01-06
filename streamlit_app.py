import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import json
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, kurtosis
import pandas as pd
import time
import cv2
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any, Union

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="MirrorGlass V3 - Detector de Fraudes em Imagens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

class AIAnalyzer:
    def __init__(self):
        self.api_key = None
        self.enabled = False
        
        if OPENAI_AVAILABLE:
            try:
                self.api_key = st.secrets.get("OPENAI_API_KEY", None)
                if self.api_key:
                    openai.api_key = self.api_key
                    self.enabled = True
            except Exception:
                pass
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_manipulation(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                'enabled': False,
                'verdict': 'DESCONHECIDO',
                'confidence': 0,
                'explanation': 'API OpenAI não configurada',
                'indicators': []
            }
        
        try:
            base64_image = self.encode_image(image)
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um especialista forense em detecção de imagens manipuladas."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analise se esta imagem é NATURAL ou MANIPULADA (editada/gerada por IA).

Responda em JSON:
{
  "verdict": "NATURAL" ou "MANIPULADA",
  "confidence": 0-100,
  "explanation": "explicação em português",
  "indicators": ["lista de indicadores"]
}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                result_json = json.loads(result_text[json_start:json_end])
            else:
                result_json = {
                    'verdict': 'DESCONHECIDO',
                    'confidence': 50,
                    'explanation': result_text,
                    'indicators': []
                }
            
            result_json['enabled'] = True
            return result_json
            
        except Exception as e:
            return {
                'enabled': True,
                'verdict': 'ERRO',
                'confidence': 0,
                'explanation': f'Erro: {str(e)}',
                'indicators': []
            }
    
    def analyze_duplicate(self, image1: Union[Image.Image, np.ndarray], 
                         image2: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                'enabled': False,
                'is_duplicate': False,
                'confidence': 0,
                'explanation': 'API não configurada',
                'relationship': 'unknown'
            }
        
        try:
            base64_image1 = self.encode_image(image1)
            base64_image2 = self.encode_image(image2)
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é especialista em análise forense de imagens."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Compare estas imagens:

São a mesma imagem/objeto/cena? Mesmo com ângulos diferentes?

JSON:
{
  "is_duplicate": true/false,
  "confidence": 0-100,
  "relationship": "identical/mirror/crop/same_object_different_angle/edited/different",
  "explanation": "explicação em português"
}"""
                            },
                            {"type": "text", "text": "Imagem 1:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image1}"
                                }
                            },
                            {"type": "text", "text": "Imagem 2:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image2}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                result_json = json.loads(result_text[json_start:json_end])
            else:
                result_json = {
                    'is_duplicate': False,
                    'confidence': 50,
                    'relationship': 'unknown',
                    'explanation': result_text
                }
            
            result_json['enabled'] = True
            return result_json
            
        except Exception as e:
            return {
                'enabled': True,
                'is_duplicate': False,
                'confidence': 0,
                'relationship': 'error',
                'explanation': f'Erro: {str(e)}'
            }

# ============================================================================
# DETECTOR ULTRA AGRESSIVO
# ============================================================================

def calcular_similaridade_completa(img1_cv, img2_cv):
    """Combina múltiplos métodos para máxima detecção"""
    try:
        scores = []
        
        # 1. Histograma de Cor (simples e efetivo)
        hist1 = cv2.calcHist([img1_cv], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2_cv], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        scores.append(max(0, hist_sim))
        
        # 2. SIFT
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            good = []
            for m_n in matches:
                if len(m_n) >= 2:
                    m, n = m_n
                    if m.distance < 0.85 * n.distance:
                        good.append(m)
            
            if len(good) >= 8:
                score_sift = min(1.0, len(good) / 30.0)
                scores.append(score_sift)
        
        # 3. ORB
        orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.1, nlevels=16)
        kp1_orb, des1_orb = orb.detectAndCompute(img1_gray, None)
        kp2_orb, des2_orb = orb.detectAndCompute(img2_gray, None)
        
        if des1_orb is not None and des2_orb is not None:
            bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
            
            good_orb = []
            for m_n in matches_orb:
                if len(m_n) >= 2:
                    m, n = m_n
                    if m.distance < 0.80 * n.distance:
                        good_orb.append(m)
            
            if len(good_orb) >= 15:
                score_orb = min(1.0, len(good_orb) / 80.0)
                scores.append(score_orb)
        
        # 4. Template Matching em diferentes escalas
        h1, w1 = img1_gray.shape
        h2, w2 = img2_gray.shape
        
        if h1 > 100 and w1 > 100 and h2 > 100 and w2 > 100:
            # Redimensionar para mesma escala
            target_size = (400, 300)
            img1_resized = cv2.resize(img1_gray, target_size)
            img2_resized = cv2.resize(img2_gray, target_size)
            
            # Matching
            result = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores.append(max(0, max_val))
        
        # 5. Média ponderada
        if len(scores) == 0:
            return 0
        
        final_score = np.mean(scores)
        
        # Boost se múltiplos métodos concordam
        if len(scores) >= 3 and all(s > 0.3 for s in scores):
            final_score = min(1.0, final_score * 1.3)
        
        return final_score
        
    except Exception as e:
        st.error(f"Erro: {e}")
        return 0

def detectar_espelhamento(img1_cv, img2_cv):
    """Testa todas as transformações possíveis"""
    try:
        normal = calcular_similaridade_completa(img1_cv, img2_cv)
        
        # Horizontal flip
        img2_h = cv2.flip(img2_cv, 1)
        flip_h = calcular_similaridade_completa(img1_cv, img2_h)
        
        # Vertical flip
        img2_v = cv2.flip(img2_cv, 0)
        flip_v = calcular_similaridade_completa(img1_cv, img2_v)
        
        # Both
        img2_both = cv2.flip(img2_cv, -1)
        flip_both = calcular_similaridade_completa(img1_cv, img2_both)
        
        scores = {
            'normal': normal,
            'horizontal': flip_h,
            'vertical': flip_v,
            'duplo': flip_both
        }
        
        max_tipo = max(scores, key=scores.get)
        max_score = scores[max_tipo]
        
        return max_score, max_tipo if max_tipo != 'normal' else None
        
    except:
        return 0, None

# ============================================================================
# TEXTURE ANALYZER (simplificado)
# ============================================================================

class TextureAnalyzer:
    def __init__(self, P=8, R=1, block_size=16, threshold=0.35):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
    
    def analyze_texture_variance(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        h, w = img_gray.shape
        block_size = self.block_size
        rows = max(1, h // block_size)
        cols = max(1, w // block_size)
        
        variance_map = np.zeros((rows, cols))
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block_lbp = lbp[i:i+block_size, j:j+block_size]
                row_idx = i // block_size
                col_idx = j // block_size
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = np.var(block_lbp) / 255.0
        
        naturalness_map = variance_map
        norm_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        suspicious_mask = norm_map < self.threshold
        naturalness_score = int(np.mean(norm_map) * 100)
        
        heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return {
            "naturalness_map": norm_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap
        }
    
    def generate_visual_report(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        naturalness_map = analysis_results["naturalness_map"]
        score = analysis_results["naturalness_score"]
        
        height, width = image.shape[:2]
        naturalness_map_resized = cv2.resize(naturalness_map, (width, height))
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        return overlay, heatmap
    
    def analyze_image(self, image):
        try:
            analysis_results = self.analyze_texture_variance(image)
            visual_report, heatmap = self.generate_visual_report(image, analysis_results)
            
            score = analysis_results["naturalness_score"]
            
            if score <= 45:
                category = "Alta chance de manipulação"
            elif score <= 70:
                category = "Textura suspeita"
            else:
                category = "Textura natural"
            
            return {
                "score": score,
                "category": category,
                "visual_report": visual_report,
                "heatmap": heatmap
            }
        except Exception as e:
            return {
                "score": 0,
                "category": "Erro",
                "visual_report": None,
                "heatmap": None
            }

# ============================================================================
# INTERFACE
# ============================================================================

st.title("🔍 MirrorGlass V3 - Detector de Fraudes")

with st.sidebar:
    st.title("⚙️ Configurações")
    
    ai_analyzer = AIAnalyzer()
    
    st.markdown("---")
    if ai_analyzer.enabled:
        st.success("✅ OpenAI API Ativa")
    else:
        st.warning("⚠️ OpenAI API Desativada")
    
    st.markdown("---")
    
    modo_analise = st.radio(
        "Modo de Análise",
        ["Duplicidade", "Manipulação por IA", "Análise Completa"],
        help="Escolha o tipo de análise"
    )
    
    if modo_analise in ["Duplicidade", "Análise Completa"]:
        st.subheader("Detecção de Duplicidade")
        
        limiar_similaridade = st.slider(
            "Limiar (%)", 
            min_value=10, 
            max_value=100, 
            value=20,
            help="Quanto menor, mais sensível"
        )
        limiar_similaridade = limiar_similaridade / 100
        
        usar_ia_duplicidade = st.checkbox(
            "Usar IA",
            value=ai_analyzer.enabled,
            disabled=not ai_analyzer.enabled
        )
    
    if modo_analise in ["Manipulação por IA", "Análise Completa"]:
        st.subheader("Detecção de Manipulação")
        
        tamanho_bloco = st.slider("Tamanho Bloco", 8, 32, 16, 4)
        threshold_lbp = st.slider("Sensibilidade", 0.1, 0.5, 0.35, 0.05)
        
        usar_ia_manipulacao = st.checkbox(
            "Usar IA",
            value=ai_analyzer.enabled,
            disabled=not ai_analyzer.enabled
        )

st.markdown("### 📤 Upload de Imagens")
uploaded_files = st.file_uploader(
    "Arraste suas imagens",
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagem(ns) carregada(s)")
    
    if st.button("🚀 Iniciar Análise", type="primary"):
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro: {arquivo.name}: {e}")
        
        # DUPLICIDADE
        if modo_analise in ["Duplicidade", "Análise Completa"]:
            st.markdown("---")
            st.markdown("## 🔄 Análise de Duplicidade")
            
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                duplicatas = {}
                total_comp = len(imagens) * (len(imagens) - 1) // 2
                comp_atual = 0
                
                for i in range(len(imagens)):
                    similares = []
                    
                    for j in range(i + 1, len(imagens)):
                        comp_atual += 1
                        progress_bar.progress(comp_atual / total_comp)
                        status_text.text(f"Comparando {nomes[i]} vs {nomes[j]}")
                        
                        img1_cv = np.array(imagens[i])[:, :, ::-1].copy()
                        img2_cv = np.array(imagens[j])[:, :, ::-1].copy()
                        
                        # Análise técnica completa
                        sim_tecnica, tipo_espelho = detectar_espelhamento(img1_cv, img2_cv)
                        
                        st.write(f"DEBUG: {nomes[i]} vs {nomes[j]} = {sim_tecnica:.3f}")
                        
                        # IA
                        ai_result = None
                        if usar_ia_duplicidade and ai_analyzer.enabled and sim_tecnica > 0.15:
                            ai_result = ai_analyzer.analyze_duplicate(imagens[i], imagens[j])
                        
                        # Combinar
                        if ai_result and ai_result.get('enabled'):
                            ai_conf = ai_result.get('confidence', 0) / 100
                            sim_final = (sim_tecnica * 0.4) + (ai_conf * 0.6)
                            
                            if sim_final >= limiar_similaridade or ai_result.get('is_duplicate'):
                                similares.append((
                                    j,
                                    sim_final,
                                    ai_result.get('relationship', 'unknown'),
                                    ai_result.get('explanation', ''),
                                    tipo_espelho
                                ))
                        else:
                            if sim_tecnica >= limiar_similaridade:
                                similares.append((
                                    j,
                                    sim_tecnica,
                                    'mirror' if tipo_espelho else 'similar',
                                    f'Técnica: {sim_tecnica:.2%}',
                                    tipo_espelho
                                ))
                    
                    if similares:
                        duplicatas[i] = similares
                
                progress_bar.empty()
                status_text.empty()
                
                if duplicatas:
                    total = sum(len(s) for s in duplicatas.values())
                    st.metric("Duplicatas Encontradas", total)
                    
                    for idx, (img_orig_idx, similares) in enumerate(duplicatas.items()):
                        st.write("---")
                        st.subheader(f"Grupo #{idx+1}")
                        
                        cols = st.columns(min(len(similares) + 1, 3))
                        
                        with cols[0]:
                            st.image(imagens[img_orig_idx], caption=nomes[img_orig_idx])
                        
                        for i, sim_data in enumerate(similares):
                            col_idx = (i + 1) % len(cols)
                            if col_idx == 0 and i > 0:
                                cols = st.columns(min(len(similares) - i + 1, 3))
                            
                            with cols[col_idx]:
                                st.image(imagens[sim_data[0]])
                                st.caption(nomes[sim_data[0]])
                                st.metric("Similaridade", f"{sim_data[1]:.2%}")
                                
                                if len(sim_data) > 4 and sim_data[4]:
                                    st.warning(f"🪞 ESP: {sim_data[4]}")
                                
                                if len(sim_data) > 2:
                                    st.info(f"🔍 {sim_data[2]}")
                                
                                if len(sim_data) > 3 and sim_data[3]:
                                    with st.expander("💡 Análise"):
                                        st.write(sim_data[3])
                else:
                    st.info("Nenhuma duplicata encontrada")
                    
            except Exception as e:
                st.error(f"Erro: {e}")
        
        # MANIPULAÇÃO
        if modo_analise in ["Manipulação por IA", "Análise Completa"]:
            st.markdown("---")
            st.markdown("## 🤖 Análise de Manipulação")
            
            try:
                analyzer = TextureAnalyzer(block_size=tamanho_bloco, threshold=threshold_lbp)
                
                for i, img in enumerate(imagens):
                    st.write("---")
                    st.subheader(nomes[i])
                    
                    report = analyzer.analyze_image(img)
                    
                    ai_result = None
                    if usar_ia_manipulacao and ai_analyzer.enabled:
                        ai_result = ai_analyzer.analyze_manipulation(img)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if report["visual_report"] is not None:
                            st.image(report["visual_report"])
                        
                        score = report['score']
                        if score <= 45:
                            st.error(f"🔴 MANIPULADA ({score}%)")
                        elif score <= 70:
                            st.warning(f"🟡 SUSPEITA ({score}%)")
                        else:
                            st.success(f"🟢 NATURAL ({score}%)")
                    
                    with col2:
                        if report["heatmap"] is not None:
                            st.image(report["heatmap"])
                        
                        if ai_result and ai_result.get('enabled'):
                            with st.expander("💡 Análise IA"):
                                st.write(ai_result.get('explanation', ''))
                                if ai_result.get('indicators'):
                                    st.write("**Indicadores:**")
                                    for ind in ai_result['indicators']:
                                        st.write(f"- {ind}")
                
            except Exception as e:
                st.error(f"Erro: {e}")

else:
    st.info("👆 Faça upload de imagens")

st.markdown("---")
st.markdown("**MirrorGlass V3** | Janeiro 2026")

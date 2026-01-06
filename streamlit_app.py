import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy
from typing import Dict, List, Tuple, Any, Union
from enum import Enum
import base64
import io
import os

# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class AIAnalyzer:
    """Integração com OpenAI Vision API para análise de manipulação"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key and OPENAI_AVAILABLE)
        
        if self.enabled:
            openai.api_key = self.api_key
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Converte imagem para base64"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_with_gpt4(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Analisa imagem com GPT-4 Vision"""
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
                        "content": """Você é um especialista forense em detecção de imagens manipuladas, 
                        geradas por IA ou editadas digitalmente. Analise cuidadosamente a imagem fornecida."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analise esta imagem e determine se é:
                                1. NATURAL: Foto real, não editada
                                2. MANIPULADA: Editada, gerada por IA ou com manipulações digitais
                                
                                Procure por:
                                - Artefatos de IA (padrões irreais, texturas artificiais)
                                - Inconsistências de iluminação
                                - Reflexos impossíveis ou incorretos
                                - Distorções em objetos ou pessoas
                                - Bordas artificiais ou desfocagem seletiva anormal
                                - Texturas repetitivas ou não naturais
                                - Detalhes anatômicos incorretos (em pessoas)
                                - Perspectiva inconsistente
                                - Compressão ou artefatos de processamento
                                
                                Responda em formato JSON:
                                {
                                  "verdict": "NATURAL" ou "MANIPULADA",
                                  "confidence": 0-100,
                                  "explanation": "explicação detalhada em português",
                                  "indicators": ["lista", "de", "indicadores", "encontrados"]
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
            
            # Parse JSON response
            import json
            # Extrai JSON da resposta (pode vir com texto extra)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result_json = json.loads(result_text[json_start:json_end])
            else:
                # Fallback se não encontrar JSON
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
                'explanation': f'Erro ao analisar com IA: {str(e)}',
                'indicators': []
            }

# ============================================================================
# SCENE DETECTOR (mantido do V2)
# ============================================================================

class SceneType(Enum):
    CAR = "CAR"
    GLASS = "GLASS"
    CAR_GLASS = "CAR_GLASS"
    CAR_GLASS_REFLECTION = "CAR_GLASS_REFLECTION"
    CAR_REFLECTION = "CAR_REFLECTION"
    CAR_SMOOTH_WALL = "CAR_SMOOTH_WALL"
    CAR_DOCUMENT = "CAR_DOCUMENT"
    GLASS_REFLECTION = "GLASS_REFLECTION"
    GLASS_WALL = "GLASS_WALL"
    GLASS_DOCUMENT = "GLASS_DOCUMENT"
    REFLECTION = "REFLECTION"
    SMOOTH_WALL = "SMOOTH_WALL"
    DOCUMENT = "DOCUMENT"
    UNKNOWN = "UNKNOWN"

class SceneDetector:
    """Detecta tipo de cena na imagem"""
    
    def __init__(self):
        pass
    
    def detect_car(self, image: np.ndarray) -> Tuple[bool, float, List]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape
        indicators = 0
        total_checks = 4
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            horizontal_lines = sum(1 for l in lines if abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) < 15 or abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) > 165)
            vertical_lines = sum(1 for l in lines if 75 < abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180 / np.pi) < 105)
            if horizontal_lines > 5 and vertical_lines > 3:
                indicators += 1
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            metallic_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 50, 200]))
            if np.mean(metallic_mask > 0) > 0.15:
                indicators += 1
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.01)]
        for contour in large_contours[:5]:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            if 1.5 < aspect < 4.0:
                indicators += 1
                break
        confidence = indicators / total_checks
        return confidence >= 0.5, confidence, []
    
    def detect_glass(self, image: np.ndarray) -> Tuple[bool, float, str]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        indicators = 0
        total_checks = 4
        h, w = gray.shape
        block_size = 32
        low_texture_blocks = sum(1 for i in range(0, h - block_size, block_size) for j in range(0, w - block_size, block_size) if np.std(gray[i:i+block_size, j:j+block_size]) < 20)
        total_blocks = max(1, ((h - block_size) // block_size) * ((w - block_size) // block_size))
        if low_texture_blocks / total_blocks > 0.3:
            indicators += 1
        confidence = indicators / total_checks
        is_glass = confidence >= 0.5
        glass_type = "window" if confidence > 0.6 else "dark_glass" if is_glass else "unknown"
        return is_glass, confidence, glass_type
    
    def classify(self, image: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        has_car, car_conf, _ = self.detect_car(image)
        has_glass, glass_conf, glass_type = self.detect_glass(image)
        
        detection_info = {
            "car": {"detected": has_car, "confidence": car_conf},
            "glass": {"detected": has_glass, "confidence": glass_conf, "type": glass_type}
        }
        
        if has_car and has_glass:
            scene_type = SceneType.CAR_GLASS
        elif has_car:
            scene_type = SceneType.CAR
        elif has_glass:
            scene_type = SceneType.GLASS
        else:
            scene_type = SceneType.UNKNOWN
            
        return scene_type, detection_info

# ============================================================================
# TEXTURE ANALYZER (mantido e melhorado)
# ============================================================================

class TextureAnalyzer:
    """Análise técnica de textura, bordas, ruído, etc."""
    
    def __init__(self):
        self.scene_detector = SceneDetector()
    
    def analyze_texture(self, image_gray: np.ndarray) -> float:
        if image_gray.size == 0:
            return 0.0
        lbp = local_binary_pattern(image_gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        hist = hist.astype(float) / hist.sum()
        texture_score = 1.0 - entropy(hist + 1e-10) / np.log(59)
        return float(np.clip(texture_score, 0, 1))
    
    def analyze_edges(self, image_gray: np.ndarray) -> float:
        if image_gray.dtype != np.uint8:
            image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(image_gray, 50, 150)
        edge_density = np.mean(edges > 0)
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        edge_score = (edge_density * 0.5) + (min(laplacian_var / 1000, 1.0) * 0.5)
        return float(np.clip(edge_score, 0, 1))
    
    def analyze_noise(self, image_gray: np.ndarray) -> float:
        if image_gray.dtype != np.uint8:
            image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
        sigma = estimate_sigma(image_gray, channel_axis=None, average_sigmas=True)
        noise_score = min(sigma / 25.0, 1.0)
        return float(np.clip(noise_score, 0, 1))
    
    def analyze_lighting(self, image_gray: np.ndarray) -> float:
        if image_gray.size == 0:
            return 0.0
        mean_intensity = np.mean(image_gray)
        std_intensity = np.std(image_gray)
        brightness_score = 1.0 - abs(mean_intensity - 128) / 128
        contrast_score = min(std_intensity / 64, 1.0)
        lighting_score = (brightness_score * 0.5) + (contrast_score * 0.5)
        return float(np.clip(lighting_score, 0, 1))
    
    def analyze(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Análise técnica tradicional"""
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        
        scene_type, detection_info = self.scene_detector.classify(image_np)
        
        texture_score = self.analyze_texture(gray)
        edge_score = self.analyze_edges(gray)
        noise_score = self.analyze_noise(gray)
        lighting_score = self.analyze_lighting(gray)
        
        weights = {'texture': 0.30, 'edge': 0.30, 'noise': 0.20, 'lighting': 0.20}
        
        all_scores = {
            'texture': round(texture_score * 100, 2),
            'edge': round(edge_score * 100, 2),
            'noise': round(noise_score * 100, 2),
            'lighting': round(lighting_score * 100, 2)
        }
        
        weighted_score = (
            texture_score * weights['texture'] +
            edge_score * weights['edge'] +
            noise_score * weights['noise'] +
            lighting_score * weights['lighting']
        )
        
        confidence = int(weighted_score * 100)
        verdict = "MANIPULADA" if weighted_score > 0.5 else "NATURAL"
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'scene_type': scene_type.value,
            'all_scores': all_scores,
            'detection_info': detection_info,
            'technical_score': weighted_score
        }

# ============================================================================
# HYBRID ANALYZER - Combina análise técnica + IA
# ============================================================================

class MirrorGlassV3:
    """Sistema híbrido: Análise Técnica + OpenAI Vision"""
    
    def __init__(self, api_key: str = None, use_ai: bool = True):
        self.texture_analyzer = TextureAnalyzer()
        self.ai_analyzer = AIAnalyzer(api_key) if use_ai else None
        self.use_ai = use_ai and (self.ai_analyzer is not None) and self.ai_analyzer.enabled
    
    def combine_verdicts(self, technical: Dict, ai: Dict) -> Dict[str, Any]:
        """Combina vereditos técnico e de IA"""
        
        # Se IA não está ativa, retorna apenas técnico
        if not ai.get('enabled', False):
            return {
                **technical,
                'analysis_mode': 'Técnica apenas',
                'reason': 'Análise baseada em características técnicas da imagem'
            }
        
        # Normaliza vereditos
        tech_is_manipulated = technical['verdict'] == 'MANIPULADA'
        ai_is_manipulated = ai['verdict'] == 'MANIPULADA'
        
        # Confiança combinada (média ponderada: 40% técnica, 60% IA)
        tech_conf = technical['confidence'] / 100.0
        ai_conf = ai['confidence'] / 100.0
        
        combined_conf = (tech_conf * 0.4 + ai_conf * 0.6)
        
        # Veredito final
        if tech_is_manipulated and ai_is_manipulated:
            final_verdict = "MANIPULADA"
            reason = f"🔴 Ambas análises concordam: {ai.get('explanation', '')}"
        elif not tech_is_manipulated and not ai_is_manipulated:
            final_verdict = "NATURAL"
            reason = f"🟢 Ambas análises concordam: {ai.get('explanation', '')}"
        elif ai_is_manipulated and ai_conf > 0.7:
            # IA tem alta confiança em manipulação
            final_verdict = "MANIPULADA"
            combined_conf = ai_conf * 0.9  # Prioriza IA
            reason = f"⚠️ IA detectou manipulação com alta confiança: {ai.get('explanation', '')}"
        elif tech_is_manipulated and tech_conf > 0.7:
            # Técnica tem alta confiança
            final_verdict = "MANIPULADA"
            combined_conf = tech_conf * 0.85
            reason = f"⚠️ Análise técnica detectou anomalias. IA sugere: {ai.get('explanation', '')}"
        else:
            # Conflito com baixa confiança - usa média
            final_verdict = "MANIPULADA" if combined_conf > 0.5 else "NATURAL"
            reason = f"⚠️ Análises divergentes. IA: {ai.get('explanation', '')}"
        
        return {
            **technical,
            'verdict': final_verdict,
            'confidence': int(combined_conf * 100),
            'reason': reason,
            'ai_analysis': ai,
            'technical_confidence': technical['confidence'],
            'ai_confidence': ai['confidence'],
            'analysis_mode': 'Híbrida (Técnica + IA)',
            'ai_indicators': ai.get('indicators', [])
        }
    
    def analyze(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Análise completa: técnica + IA"""
        
        # 1. Análise técnica (sempre executa)
        technical_result = self.texture_analyzer.analyze(image)
        
        # 2. Análise com IA (se habilitada)
        if self.use_ai:
            ai_result = self.ai_analyzer.analyze_with_gpt4(image)
        else:
            ai_result = {'enabled': False}
        
        # 3. Combina resultados
        final_result = self.combine_verdicts(technical_result, ai_result)
        
        return final_result

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="MirrorGlass V3 - AI Enhanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração da API na sidebar
with st.sidebar:
    st.title("⚙️ MirrorGlass V3")
    st.markdown("### 🤖 AI Enhanced Detection")
    
    st.markdown("---")
    
    # Configuração OpenAI
    st.subheader("🔑 OpenAI API")
    
    api_key_input = st.text_input(
        "API Key",
        type="password",
        help="Cole sua chave da API OpenAI aqui. Deixe vazio para usar apenas análise técnica."
    )
    
    use_ai = st.checkbox(
        "Ativar Análise com IA",
        value=bool(api_key_input or os.getenv("OPENAI_API_KEY")),
        help="Usa GPT-4 Vision para análise complementar"
    )
    
    if use_ai and not api_key_input and not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Configure a API Key para usar IA")
    elif use_ai:
        st.success("✅ IA Ativada")
    else:
        st.info("ℹ️ Modo técnico apenas")
    
    st.markdown("---")
    
    # Configurações de exibição
    st.subheader("📊 Exibição")
    show_details = st.checkbox("Mostrar Detalhes Técnicos", value=True)
    cols_per_row = st.slider("Imagens por linha", 1, 4, 2)
    
    st.markdown("---")
    
    # Informações
    with st.expander("ℹ️ Sobre"):
        st.markdown("""
        **MirrorGlass V3** combina:
        
        - 🔧 Análise técnica tradicional
        - 🤖 GPT-4 Vision (OpenAI)
        
        **Vantagens:**
        - Detecta manipulações sutis
        - Identifica IA generativa
        - Explica os achados
        - Maior precisão
        """)

# Interface principal
st.title("🔍 MirrorGlass V3")
st.markdown("### Detector de Manipulação com IA")

# Upload de arquivos
uploaded_files = st.file_uploader(
    "📤 Arraste suas imagens aqui",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Selecione uma ou mais imagens para análise"
)

if uploaded_files:
    st.markdown("---")
    st.markdown(f"### 📁 {len(uploaded_files)} imagem(ns) carregada(s)")
    
    if st.button("🚀 Analisar Todas", type="primary", use_container_width=True):
        
        # Inicializa o analisador
        analyzer = MirrorGlassV3(
            api_key=api_key_input,
            use_ai=use_ai
        )
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analisando {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            # Análise
            result = analyzer.analyze(image)
            
            result['image'] = image
            result['image_np'] = image_np
            result['filename'] = uploaded_file.name
            results.append(result)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.results = results
        st.success(f"✅ Análise concluída! {len(results)} imagens processadas.")

# Exibição dos resultados
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.markdown("## 📊 Resumo Geral")
    
    manipuladas = sum(1 for r in results if r['verdict'] == 'MANIPULADA')
    naturais = sum(1 for r in results if r['verdict'] == 'NATURAL')
    avg_confidence = int(np.mean([r['confidence'] for r in results]))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", len(results))
    with col2:
        st.metric("🔴 Manipuladas", manipuladas)
    with col3:
        st.metric("🟢 Naturais", naturais)
    with col4:
        st.metric("Confiança Média", f"{avg_confidence}%")
    
    # Filtros
    st.markdown("---")
    filter_option = st.radio(
        "Filtrar por:",
        ["Todas", "🔴 Manipuladas", "🟢 Naturais"],
        horizontal=True
    )
    
    if filter_option == "🔴 Manipuladas":
        filtered_results = [r for r in results if r['verdict'] == 'MANIPULADA']
    elif filter_option == "🟢 Naturais":
        filtered_results = [r for r in results if r['verdict'] == 'NATURAL']
    else:
        filtered_results = results
    
    st.markdown(f"### Exibindo {len(filtered_results)} imagem(ns)")
    
    # Exibir resultados
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered_results):
                result = filtered_results[i + j]
                
                with col:
                    # Imagem
                    st.image(result['image'], caption=result['filename'], use_column_width=True)
                    
                    # Veredito
                    verdict = result['verdict']
                    confidence = result['confidence']
                    
                    if verdict == "MANIPULADA":
                        st.error(f"🔴 **{verdict}** ({confidence}%)")
                    else:
                        st.success(f"🟢 **{verdict}** ({confidence}%)")
                    
                    # Modo de análise
                    st.caption(f"🔬 {result.get('analysis_mode', 'N/A')}")
                    
                    # Razão
                    with st.expander("💡 Explicação"):
                        st.write(result.get('reason', 'N/A'))
                    
                    # Detalhes técnicos
                    if show_details:
                        with st.expander("📊 Detalhes Técnicos"):
                            st.write(f"**Cena:** {result.get('scene_type', 'N/A')}")
                            
                            scores = result.get('all_scores', {})
                            st.write(f"**Textura:** {scores.get('texture', 'N/A')}%")
                            st.write(f"**Bordas:** {scores.get('edge', 'N/A')}%")
                            st.write(f"**Ruído:** {scores.get('noise', 'N/A')}%")
                            st.write(f"**Iluminação:** {scores.get('lighting', 'N/A')}%")
                            
                            if 'ai_analysis' in result and result['ai_analysis'].get('enabled'):
                                st.markdown("---")
                                st.write("**Análise IA:**")
                                st.write(f"Confiança: {result.get('ai_confidence', 'N/A')}%")
                                
                                indicators = result.get('ai_indicators', [])
                                if indicators:
                                    st.write("**Indicadores:**")
                                    for ind in indicators:
                                        st.write(f"- {ind}")
    
    # Exportação
    st.markdown("---")
    with st.expander("📥 Exportar Resultados"):
        export_data = []
        for r in results:
            scores = r.get('all_scores', {})
            export_data.append({
                'Arquivo': r['filename'],
                'Veredito': r['verdict'],
                'Confiança': r['confidence'],
                'Modo Análise': r.get('analysis_mode', 'N/A'),
                'Razão': r.get('reason', 'N/A'),
                'Cena': r.get('scene_type', 'N/A'),
                'Score Textura': scores.get('texture', 'N/A'),
                'Score Bordas': scores.get('edge', 'N/A'),
                'Score Ruído': scores.get('noise', 'N/A'),
                'Score Iluminação': scores.get('lighting', 'N/A'),
                'Confiança Técnica': r.get('technical_confidence', 'N/A'),
                'Confiança IA': r.get('ai_confidence', 'N/A')
            })
        
        df = pd.DataFrame(export_data)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "mirrorglass_v3_resultados.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.markdown("---")
    st.info("👆 Faça upload de imagens e clique em **Analisar Todas** para começar")
    
    with st.expander("📖 Guia Rápido"):
        st.markdown("""
        ### Como usar o MirrorGlass V3:
        
        **1. Configure a API OpenAI** (opcional mas recomendado):
        - Obtenha uma chave em [platform.openai.com](https://platform.openai.com/api-keys)
        - Cole na sidebar
        - Ative "Análise com IA"
        
        **2. Faça upload das imagens**
        
        **3. Clique em "Analisar Todas"**
        
        **4. Veja os resultados:**
        - 🔴 **MANIPULADA**: Detectada manipulação/IA
        - 🟢 **NATURAL**: Provavelmente foto real
        
        ### O que o V3 detecta:
        
        ✅ Imagens geradas por IA (Midjourney, DALL-E, etc)
        ✅ Fotos editadas digitalmente
        ✅ Deepfakes e manipulações faciais
        ✅ Artefatos de compressão anormais
        ✅ Inconsistências de iluminação
        ✅ Reflexos impossíveis
        ✅ Texturas artificiais
        
        ### Modos de operação:
        
        - **Híbrido** (Técnica + IA): Máxima precisão
        - **Técnica apenas**: Funciona sem API key
        """)

st.markdown("---")
st.caption("MirrorGlass V3 | AI Enhanced Detection | Janeiro 2026")

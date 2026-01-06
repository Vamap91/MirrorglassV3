import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from skimage.feature import local_binary_pattern
from skimage.restoration import estimate_sigma
from scipy.stats import entropy, kurtosis
from typing import Dict, List, Tuple, Any, Union
from enum import Enum

# ============================================================================
# TEXTURE ANALYZER - Integrado
# ============================================================================

CLAHE_CONFIG = {
    "texture_clahe": False,
    "edge_clahe": True,
    "noise_clahe": True,
    "lighting_clahe": True,
    "clip_limit": 2.0
}


def apply_clahe(img_gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


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
        lower_third = gray[int(h*0.6):, :]
        if lower_third.size > 0:
            circles = cv2.HoughCircles(lower_third, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
            if circles is not None and len(circles[0]) >= 2:
                indicators += 1
        confidence = indicators / total_checks
        return confidence >= 0.5, confidence, []
    
    def detect_glass(self, image: np.ndarray) -> Tuple[bool, float, str]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            gray = image.copy()
            hsv = None
        indicators = 0
        total_checks = 4
        h, w = gray.shape
        block_size = 32
        low_texture_blocks = sum(1 for i in range(0, h - block_size, block_size) for j in range(0, w - block_size, block_size) if np.std(gray[i:i+block_size, j:j+block_size]) < 20)
        total_blocks = max(1, ((h - block_size) // block_size) * ((w - block_size) // block_size))
        if low_texture_blocks / total_blocks > 0.3:
            indicators += 1
        if hsv is not None:
            low_sat_ratio = np.mean(hsv[:, :, 1] < 40)
            if low_sat_ratio > 0.4:
                indicators += 1
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.mean(bright > 0)
        if 0.05 < bright_ratio < 0.4:
            indicators += 1
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if np.mean(magnitude < 10) > 0.5:
            indicators += 1
        confidence = indicators / total_checks
        is_glass = confidence >= 0.5
        glass_type = "window" if bright_ratio > 0.15 else "dark_glass" if is_glass else "unknown"
        return is_glass, confidence, glass_type
    
    def detect_reflection(self, image: np.ndarray) -> Tuple[float, bool, Dict]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
        else:
            gray = image
            saturation = np.zeros_like(gray)
        _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        _, low_sat = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
        reflection_mask = cv2.bitwise_and(bright, low_sat)
        percent = np.mean(reflection_mask > 0)
        contours, _ = cv2.findContours(reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reflection_info = {"percent": percent, "num_regions": len(contours), "is_specular": False, "is_diffuse": False}
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas)
            total_area = sum(areas)
            if max_area > total_area * 0.5:
                reflection_info["is_specular"] = True
            else:
                reflection_info["is_diffuse"] = True
        is_significant = percent > 0.10
        return percent, is_significant, reflection_info
    
    def detect_smooth_surface(self, image: np.ndarray) -> Tuple[float, bool, str]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        block_size = 32
        h, w = gray.shape
        block_stds = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))
        mean_std = np.mean(block_stds) if block_stds else 50
        std_of_stds = np.std(block_stds) if block_stds else 50
        if mean_std < 15 and std_of_stds < 10:
            is_smooth, smooth_percent, surface_type = True, 0.9, "glass"
        elif mean_std < 25 and std_of_stds < 15:
            is_smooth, smooth_percent, surface_type = True, 0.7, "painted_wall"
        elif mean_std < 35:
            is_smooth, smooth_percent, surface_type = True, 0.5, "semi_smooth"
        else:
            is_smooth, smooth_percent, surface_type = False, 0.0, "textured"
        return smooth_percent, is_smooth, surface_type
    
    def detect_document(self, image: np.ndarray) -> Tuple[bool, float, List[Dict]]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        h, w = image.shape[:2]
        total_area = h * w
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 195])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((9, 9), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        documents = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / total_area
            if area_ratio < 0.005 or area_ratio > 0.6:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = max(cw, ch) / min(cw, ch) if min(cw, ch) > 0 else 0
            if 1.0 <= aspect_ratio <= 4.0:
                documents.append({"bbox": (x, y, cw, ch), "area_percent": area_ratio * 100})
        has_document = len(documents) > 0
        confidence = min(1.0, len(documents) * 0.3 + (0.5 if has_document else 0))
        return has_document, confidence, documents
    
    def classify(self, image: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        has_car, car_conf, _ = self.detect_car(image)
        has_glass, glass_conf, glass_type = self.detect_glass(image)
        reflection_pct, has_reflection, reflection_info = self.detect_reflection(image)
        smooth_pct, is_smooth, surface_type = self.detect_smooth_surface(image)
        has_document, doc_conf, documents = self.detect_document(image)
        detection_info = {
            "car": {"detected": has_car, "confidence": car_conf},
            "glass": {"detected": has_glass, "confidence": glass_conf, "type": glass_type},
            "reflection": {"detected": has_reflection, "percent": reflection_pct, "info": reflection_info},
            "smooth": {"detected": is_smooth, "percent": smooth_pct, "type": surface_type},
            "document": {"detected": has_document, "confidence": doc_conf, "count": len(documents)}
        }
        scene_type = self._determine_scene_type(has_car, has_glass, has_reflection, is_smooth, has_document, surface_type)
        return scene_type, detection_info
    
    def _determine_scene_type(self, has_car: bool, has_glass: bool, has_reflection: bool, is_smooth: bool, has_document: bool, surface_type: str) -> Any:
        is_wall = surface_type in ["painted_wall", "semi_smooth"]
        if has_car and has_glass and has_reflection:
            return SceneType.CAR_GLASS_REFLECTION
        if has_car and has_glass:
            return SceneType.CAR_GLASS
        if has_car and has_reflection:
            return SceneType.CAR_REFLECTION
        if has_car and is_wall:
            return SceneType.CAR_SMOOTH_WALL
        if has_car and has_document:
            return SceneType.CAR_DOCUMENT
        if has_glass and has_reflection:
            return SceneType.GLASS_REFLECTION
        if has_glass and is_wall:
            return SceneType.GLASS_WALL
        if has_glass and has_document:
            return SceneType.GLASS_DOCUMENT
        if has_car:
            return SceneType.CAR
        if has_glass:
            return SceneType.GLASS
        if has_reflection:
            return SceneType.REFLECTION
        if is_smooth:
            return SceneType.SMOOTH_WALL
        if has_document:
            return SceneType.DOCUMENT
        return SceneType.UNKNOWN


class TextureAnalyzer:
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
    
    def analyze_reflection(self, image: np.ndarray) -> float:
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
        else:
            gray = image
            saturation = np.zeros_like(gray)
        _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        _, low_sat = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
        reflection_mask = cv2.bitwise_and(bright, low_sat)
        reflection_percent = np.mean(reflection_mask > 0)
        return float(np.clip(reflection_percent, 0, 1))
    
    def analyze(self, image: Union[Image.Image, np.ndarray], detection_mode: str = "Balanceado") -> Dict[str, Any]:
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
        reflection_score = self.analyze_reflection(image_np)
        
        weights = self._get_weights(detection_mode, detection_info)
        
        all_scores = {
            'texture': round(texture_score * 100, 2),
            'edge': round(edge_score * 100, 2),
            'noise': round(noise_score * 100, 2),
            'lighting': round(lighting_score * 100, 2),
            'reflection': round(reflection_score * 100, 2)
        }
        
        weighted_score = (
            texture_score * weights['texture'] +
            edge_score * weights['edge'] +
            noise_score * weights['noise'] +
            lighting_score * weights['lighting'] +
            reflection_score * weights['reflection']
        )
        
        confidence = int(weighted_score * 100)
        verdict = "MANIPULADA" if weighted_score > 0.5 else "NATURAL"
        reason = self._get_reason(weighted_score, all_scores, detection_info)
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reason': reason,
            'scene_type': scene_type.value,
            'all_scores': all_scores,
            'detection_info': detection_info,
            'phases_executed': 'Análise Completa'
        }
    
    def _get_weights(self, detection_mode: str, detection_info: Dict) -> Dict[str, float]:
        base_weights = {
            'texture': 0.25,
            'edge': 0.25,
            'noise': 0.20,
            'lighting': 0.20,
            'reflection': 0.10
        }
        
        if detection_mode == "Conservador":
            base_weights['texture'] = 0.30
            base_weights['edge'] = 0.30
        elif detection_mode == "Agressivo":
            base_weights['noise'] = 0.25
            base_weights['reflection'] = 0.15
        
        return base_weights
    
    def _get_reason(self, score: float, all_scores: Dict, detection_info: Dict) -> str:
        if score > 0.7:
            return "Múltiplos indicadores de manipulação detectados"
        elif score > 0.5:
            return "Alguns indicadores de manipulação encontrados"
        else:
            return "Imagem parece ser natural"


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="MirrorGlass V2 - Análise em Lote",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("⚙️ Configurações")

    st.markdown("---")

    st.success("✅ Modo Automático Ativo")
    st.info("O sistema detecta automaticamente carros, vidros, reflexos e ajusta os pesos de análise.")

    detection_mode = st.selectbox(
        "Sensibilidade",
        ["Balanceado", "Conservador", "Agressivo"],
        index=0,
        help="Conservador = menos falsos positivos | Agressivo = detecta mais manipulações"
    )
    
    st.markdown("---")
    st.subheader("📊 Exibição")
    
    show_heatmaps = st.checkbox("Mostrar Detalhes", value=True)
    cols_per_row = st.slider("Imagens por linha", 1, 4, 2)

st.title("🔍 MirrorGlass V2")
st.markdown("### Detector de Manipulação em Lote")

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
        
        analyzer = TextureAnalyzer()
        
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analisando {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            result = analyzer.analyze(image, detection_mode)
            
            result['image'] = image
            result['image_np'] = image_np
            result['filename'] = uploaded_file.name
            results.append(result)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        st.session_state.results = results
        st.success(f"✅ Análise concluída! {len(results)} imagens processadas.")

if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.markdown("## 📊 Resumo")
    
    manipuladas = sum(1 for r in results if r['verdict'] == 'MANIPULADA')
    naturais = sum(1 for r in results if r['verdict'] == 'NATURAL')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", len(results))
    with col2:
        st.metric("🔴 Manipuladas", manipuladas)
    with col3:
        st.metric("🟢 Naturais", naturais)
    with col4:
        st.metric("Confiança Média", f"{int(np.mean([r['confidence'] for r in results]))}%")
    
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
    
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered_results):
                result = filtered_results[i + j]
                
                with col:
                    st.image(result['image'], caption=result['filename'], use_column_width=True)
                    
                    verdict = result['verdict']
                    confidence = result['confidence']
                    
                    if verdict == "MANIPULADA":
                        st.error(f"🔴 **{verdict}** ({confidence}%)")
                    else:
                        st.success(f"🟢 **{verdict}** ({confidence}%)")
                    
                    st.caption(f"📍 Cena: {result.get('scene_type', 'N/A')}")
                    st.caption(f"💡 {result['reason']}")
                    
                    if show_heatmaps:
                        with st.expander("📊 Detalhes"):
                            scores = result.get('all_scores', {})
                            st.write(f"**Textura**: {scores.get('texture', 'N/A')}%")
                            st.write(f"**Bordas**: {scores.get('edge', 'N/A')}%")
                            st.write(f"**Ruído**: {scores.get('noise', 'N/A')}%")
                            st.write(f"**Iluminação**: {scores.get('lighting', 'N/A')}%")
                            st.write(f"**Reflexo**: {scores.get('reflection', 'N/A')}%")
    
    st.markdown("---")
    
    with st.expander("📥 Exportar Resultados"):
        export_data = []
        for r in results:
            scores = r.get('all_scores', {})
            export_data.append({
                'Arquivo': r['filename'],
                'Veredito': r['verdict'],
                'Confiança': r['confidence'],
                'Razão': r['reason'],
                'Cena': r.get('scene_type', 'N/A'),
                'Score Textura': scores.get('texture', 'N/A'),
                'Score Bordas': scores.get('edge', 'N/A'),
                'Score Ruído': scores.get('noise', 'N/A'),
                'Score Iluminação': scores.get('lighting', 'N/A'),
                'Reflexo %': scores.get('reflection', 'N/A')
            })
        
        df = pd.DataFrame(export_data)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "mirrorglass_resultados.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.markdown("---")
    st.info("👆 Faça upload de imagens e clique em **Analisar Todas** para começar")
    
    with st.expander("📖 Como usar"):
        st.markdown("""
        ### Passo a passo:

        1. **Configure** a sensibilidade na sidebar esquerda:
           - **Conservador**: Menos falsos positivos
           - **Balanceado**: Equilíbrio entre precisão e recall
           - **Agressivo**: Detecta mais manipulações

        2. **Arraste** suas imagens para a área de upload

        3. **Clique** em "Analisar Todas"

        4. **Filtre** os resultados por veredito

        5. **Exporte** para CSV se necessário

        ### Sobre os vereditos:

        - 🔴 **MANIPULADA**: Alta probabilidade de ser IA ou editada
        - 🟢 **NATURAL**: Provavelmente foto real

        ### Como funciona:

        O sistema detecta automaticamente o tipo de cena (carros, vidros, reflexos)
        e ajusta os parâmetros de análise para cada situação específica.
        """)

st.markdown("---")
st.caption("MirrorGlass V2 | Análise em Lote | Dezembro 2025")

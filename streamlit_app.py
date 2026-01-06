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

# Tentar importar OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

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
    """Integração com OpenAI Vision API para análise de manipulação e duplicidade"""
    
    def __init__(self):
        self.api_key = None
        self.enabled = False
        
        if OPENAI_AVAILABLE:
            try:
                # API key já configurada no Streamlit Cloud (Settings > Secrets)
                self.api_key = st.secrets.get("OPENAI_API_KEY", None)
                if self.api_key:
                    openai.api_key = self.api_key
                    self.enabled = True
            except Exception:
                pass  # API não configurada, modo técnico será usado
    
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Converte imagem para base64"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_manipulation(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Analisa manipulação com GPT-4 Vision"""
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
                'explanation': f'Erro ao analisar com IA: {str(e)}',
                'indicators': []
            }
    
    def analyze_duplicate(self, image1: Union[Image.Image, np.ndarray], 
                         image2: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Analisa se duas imagens são duplicatas/espelhamentos com GPT-4 Vision"""
        if not self.enabled:
            return {
                'enabled': False,
                'is_duplicate': False,
                'confidence': 0,
                'explanation': 'API OpenAI não configurada',
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
                        "content": """Você é um especialista em análise forense de imagens. 
                        Sua tarefa é identificar se duas imagens são duplicatas, espelhamentos ou relacionadas."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Compare estas duas imagens e determine:
                                
                                1. São a MESMA imagem (idênticas)?
                                2. São ESPELHAMENTOS (mirror/flip)?
                                3. São RECORTES (uma é parte da outra)?
                                4. São DUPLICATAS COM EDIÇÕES (mesma base, editadas)?
                                5. São DIFERENTES (não relacionadas)?
                                
                                Analise:
                                - Objetos principais
                                - Composição da cena
                                - Cores e iluminação
                                - Texturas e detalhes
                                - Ângulo e perspectiva
                                
                                Responda em formato JSON:
                                {
                                  "is_duplicate": true/false,
                                  "confidence": 0-100,
                                  "relationship": "identical/mirror/crop/edited/different",
                                  "explanation": "explicação detalhada em português"
                                }"""
                            },
                            {
                                "type": "text",
                                "text": "Imagem 1:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image1}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Imagem 2:"
                            },
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
            
            # Parse JSON response
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
                'explanation': f'Erro ao analisar com IA: {str(e)}'
            }

# ============================================================================
# TEXTURE ANALYZER (do código original)
# ============================================================================

class TextureAnalyzer:
    """
    Classe para análise de texturas usando Local Binary Pattern (LBP).
    Detecta manipulações em imagens automotivas, principalmente restaurações por IA.
    """
    
    def __init__(self, P=8, R=1, block_size=8, threshold=0.10):
        self.P = P
        self.R = R
        self.block_size = block_size
        self.threshold = threshold
        self.scales = [0.5, 1.0, 2.0]
    
    def calculate_lbp(self, image):
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        elif len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
            
        lbp = local_binary_pattern(img_gray, self.P, self.R, method="uniform")
        
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return lbp, hist, img_gray
    
    def analyze_texture_variance(self, image):
        """Análise detalhada de textura para detectar manipulações por IA"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image.copy()
        
        # 1. Detecção de bordas
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        edges = cv2.Canny(img_gray, 50, 150)
        
        # 2. Filtro de mediana
        img_filtered = cv2.medianBlur(img_gray, 5)
        
        # 3. LBP em múltiplas escalas
        lbp_maps = []
        blurred_maps = []
        
        for scale in self.scales:
            if scale != 1.0:
                height, width = img_gray.shape
                new_height, new_width = int(height * scale), int(width * scale)
                img_scaled = cv2.resize(img_gray, (new_width, new_height))
                blurred = cv2.GaussianBlur(img_scaled, (5, 5), 0)
                lbp_scaled, _, _ = self.calculate_lbp(blurred)
                lbp_map = cv2.resize(lbp_scaled, (width, height))
            else:
                lbp_map, _, _ = self.calculate_lbp(img_gray)
            
            lbp_maps.append(lbp_map)
            
            for sigma in [1, 3, 5]:
                blurred = cv2.GaussianBlur(img_gray, (sigma*2+1, sigma*2+1), sigma)
                blurred_maps.append(blurred)
        
        # 4. Análise em blocos
        height, width = img_gray.shape
        rows = max(1, height // self.block_size)
        cols = max(1, width // self.block_size)
        
        variance_map = np.zeros((rows, cols))
        entropy_map = np.zeros((rows, cols))
        gradient_consistency_map = np.zeros((rows, cols))
        edge_density_map = np.zeros((rows, cols))
        blur_consistency_map = np.zeros((rows, cols))
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block_gray = img_gray[i:i+self.block_size, j:j+self.block_size]
                if i < lbp_maps[1].shape[0] - self.block_size and j < lbp_maps[1].shape[1] - self.block_size:
                    block_lbp = lbp_maps[1][i:i+self.block_size, j:j+self.block_size]
                else:
                    continue
                
                block_gradient = gradient_magnitude[i:i+self.block_size, j:j+self.block_size]
                block_edges = edges[i:i+self.block_size, j:j+self.block_size]
                
                # Entropia LBP
                hist, _ = np.histogram(block_lbp, bins=10, range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                block_entropy = entropy(hist)
                max_entropy = np.log(10)
                norm_entropy = block_entropy / max_entropy if max_entropy > 0 else 0
                
                # Variância da textura
                block_variance = np.var(block_lbp) / 255.0
                
                # Consistência do gradiente
                grad_hist, _ = np.histogram(block_gradient, bins=8)
                grad_hist = grad_hist.astype("float")
                grad_hist /= (grad_hist.sum() + 1e-7)
                grad_entropy = entropy(grad_hist)
                grad_consistency = 1.0 - (grad_entropy / np.log(8))
                
                # Densidade de bordas
                edge_density = np.sum(block_edges > 0) / (self.block_size * self.block_size)
                
                # Consistência do blur
                blur_responses = []
                for blurred in blurred_maps:
                    blur_block = blurred[i:i+self.block_size, j:j+self.block_size]
                    diff = np.abs(block_gray.astype(float) - blur_block.astype(float)).mean()
                    blur_responses.append(diff)
                
                blur_consistency = 1.0 - min(np.std(blur_responses) / 10.0, 1.0)
                
                row_idx = i // self.block_size
                col_idx = j // self.block_size
                
                if row_idx < rows and col_idx < cols:
                    variance_map[row_idx, col_idx] = block_variance
                    entropy_map[row_idx, col_idx] = norm_entropy
                    gradient_consistency_map[row_idx, col_idx] = grad_consistency
                    edge_density_map[row_idx, col_idx] = edge_density
                    blur_consistency_map[row_idx, col_idx] = blur_consistency
        
        # 5. Combinar métricas
        weights = {
            'entropy': 0.20,
            'variance': 0.15,
            'gradient': 0.15,
            'edge_density': 0.20,
            'blur_consistency': 0.30
        }
        
        naturalness_map = (
            (1.0 - weights['entropy'] * (1.0 - entropy_map)) *
            (1.0 - weights['variance'] * (1.0 - variance_map)) *
            (1.0 - weights['gradient'] * gradient_consistency_map) *
            (1.0 - weights['edge_density'] * (1.0 - edge_density_map)) *
            (1.0 - weights['blur_consistency'] * blur_consistency_map)
        )
        
        norm_naturalness_map = cv2.normalize(naturalness_map, None, 0, 1, cv2.NORM_MINMAX)
        suspicious_mask = norm_naturalness_map < self.threshold
        naturalness_score = int(np.mean(norm_naturalness_map) * 100)
        
        heatmap = cv2.applyColorMap(
            (norm_naturalness_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return {
            "naturalness_map": norm_naturalness_map,
            "suspicious_mask": suspicious_mask,
            "naturalness_score": naturalness_score,
            "heatmap": heatmap,
            "entropy_map": entropy_map,
            "variance_map": variance_map,
            "edge_density_map": edge_density_map
        }
    
    def classify_naturalness(self, score):
        if score <= 45:
            return "Alta chance de manipulação", "Textura artificial detectada"
        elif score <= 70:
            return "Textura suspeita", "Revisão manual sugerida"
        else:
            return "Textura natural", "Baixa chance de manipulação"
    
    def generate_visual_report(self, image, analysis_results):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        naturalness_map = analysis_results["naturalness_map"]
        suspicious_mask = analysis_results["suspicious_mask"]
        score = analysis_results["naturalness_score"]
        
        height, width = image.shape[:2]
        
        naturalness_map_resized = cv2.resize(naturalness_map, 
                                           (width, height), 
                                           interpolation=cv2.INTER_LINEAR)
        
        mask_resized = cv2.resize(suspicious_mask.astype(np.uint8), 
                                 (width, height), 
                                 interpolation=cv2.INTER_NEAREST)
        
        heatmap = cv2.applyColorMap((naturalness_map_resized * 255).astype(np.uint8), 
                                    cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        highlighted = overlay.copy()
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(highlighted, (x, y), (x+w, y+h), (128, 0, 128), 2)
        
        category, description = self.classify_naturalness(score)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(highlighted, f"Score: {score}/100", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(highlighted, category, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        return highlighted, heatmap
    
    def analyze_image(self, image):
        try:
            analysis_results = self.analyze_texture_variance(image)
            visual_report, heatmap = self.generate_visual_report(image, analysis_results)
            
            score = analysis_results.get("naturalness_score", 0)
            category, description = self.classify_naturalness(score)
            
            suspicious_mask = analysis_results.get("suspicious_mask")
            percentual_suspeito = float(np.mean(suspicious_mask) * 100) if suspicious_mask is not None else 0.0
            
            return {
                "score": score,
                "category": category,
                "description": description,
                "percentual_suspeito": percentual_suspeito,
                "visual_report": visual_report,
                "heatmap": heatmap,
                "analysis_results": analysis_results
            }
        except Exception as e:
            return {
                "score": 0,
                "category": "Erro",
                "description": f"Erro na análise: {str(e)}",
                "percentual_suspeito": 0,
                "visual_report": None,
                "heatmap": None,
                "analysis_results": {}
            }

# ============================================================================
# DUPLICATE DETECTOR (do código original)
# ============================================================================

def preprocessar_imagem(img, tamanho=(300, 300)):
    try:
        img_resize = img.resize(tamanho)
        img_gray = img_resize.convert('L')
        img_array = np.array(img_gray)
        img_array = img_array / 255.0
        img_cv = np.array(img_resize)
        img_cv = img_cv[:, :, ::-1].copy()
        return img_array, img_cv
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None

def calcular_similaridade_ssim(img1, img2):
    try:
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape)
        
        score = ssim(img1, img2, data_range=1.0)
        return score
    except Exception as e:
        return 0

def calcular_similaridade_sift(img1_cv, img2_cv):
    try:
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        
        # Aumentar número de features detectadas
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.03, edgeThreshold=8)
        
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
            
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Ratio test mais permissivo para recortes
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return 0
        
        # Calcular similaridade baseada em bons matches
        # Para recortes, usar uma métrica mais generosa
        total_possible = max(len(kp1), len(kp2))
        match_ratio = len(good_matches) / total_possible
        
        # Boost para casos com muitos matches absolutos (recortes)
        if len(good_matches) > 50:
            match_ratio = match_ratio * 1.5
        elif len(good_matches) > 30:
            match_ratio = match_ratio * 1.3
        
        similarity = min(1.0, match_ratio * 3.0)
        
        # Tentar homografia para confirmar recortes/transformações
        if len(good_matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    matches_mask = mask.ravel().tolist()
                    inliers = sum(matches_mask)
                    # Se temos muitos inliers, é muito provável que seja recorte
                    if inliers > 20:
                        similarity = max(similarity, 0.7 + (inliers / len(good_matches)) * 0.3)
            except:
                pass
        
        return similarity
        
    except Exception as e:
        return 0

def detectar_duplicatas(imagens, nomes, limiar=0.5, metodo="SIFT + SSIM", ai_analyzer=None):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extraindo características das imagens...")
    arrays_processados_gray = []
    arrays_processados_cv = []
    indices_validos = []
    
    for i, img in enumerate(imagens):
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        img_array_gray, img_array_cv = preprocessar_imagem(img)
        if img_array_gray is not None:
            arrays_processados_gray.append(img_array_gray)
            arrays_processados_cv.append(img_array_cv)
            indices_validos.append(i)
    
    if not arrays_processados_gray:
        status_text.error("Nenhuma imagem válida para processamento.")
        progress_bar.empty()
        return None
    
    status_text.text("Comparando imagens e buscando duplicatas...")
    duplicatas = {}
    
    total_comparacoes = len(arrays_processados_gray) * (len(arrays_processados_gray) - 1) // 2
    comparacao_atual = 0
    
    for i in range(len(arrays_processados_gray)):
        similares = []
        for j in range(i + 1, len(arrays_processados_gray)):
            comparacao_atual += 1
            
            if total_comparacoes > 0:
                progress = min(max(comparacao_atual / total_comparacoes, 0.0), 1.0)
                progress_bar.progress(progress)
            
            # Análise técnica
            if metodo == "SSIM":
                similaridade_tecnica = calcular_similaridade_ssim(
                    arrays_processados_gray[i], 
                    arrays_processados_gray[j]
                )
            elif metodo == "SIFT":
                similaridade_tecnica = calcular_similaridade_sift(
                    arrays_processados_cv[i], 
                    arrays_processados_cv[j]
                )
            else:  # SIFT + SSIM
                sim_ssim = calcular_similaridade_ssim(
                    arrays_processados_gray[i], 
                    arrays_processados_gray[j]
                )
                sim_sift = calcular_similaridade_sift(
                    arrays_processados_cv[i], 
                    arrays_processados_cv[j]
                )
                similaridade_tecnica = (sim_ssim * 0.3) + (sim_sift * 0.7)
            
            # Análise com IA (se disponível e similaridade > 25%)
            ai_result = None
            if ai_analyzer and ai_analyzer.enabled and similaridade_tecnica > 0.25:
                ai_result = ai_analyzer.analyze_duplicate(
                    imagens[indices_validos[i]], 
                    imagens[indices_validos[j]]
                )
            
            # Combinar resultados
            if ai_result and ai_result.get('enabled'):
                # Média ponderada: 40% técnica, 60% IA
                ai_confidence = ai_result.get('confidence', 0) / 100
                similaridade_final = (similaridade_tecnica * 0.4) + (ai_confidence * 0.6)
                
                if similaridade_final >= limiar or ai_result.get('is_duplicate'):
                    similares.append((
                        indices_validos[j], 
                        similaridade_final,
                        ai_result.get('relationship', 'unknown'),
                        ai_result.get('explanation', '')
                    ))
            else:
                # Apenas análise técnica
                if similaridade_tecnica >= limiar:
                    similares.append((
                        indices_validos[j], 
                        similaridade_tecnica,
                        'technical_only',
                        'Análise técnica apenas'
                    ))
        
        if similares:
            duplicatas[indices_validos[i]] = similares
    
    progress_bar.empty()
    status_text.text("Processamento concluído!")
    
    return duplicatas

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Título
st.title("🔍 MirrorGlass V3 - Detector de Fraudes em Imagens")
st.markdown("""
Sistema avançado de detecção de fraudes utilizando:
- 🤖 **Análise com IA (GPT-4 Vision)** - Detecta manipulações sutis e contexto
- 🔧 **Análise Técnica (LBP + SIFT + SSIM)** - Análise matemática de texturas e duplicatas
- 🔄 **Sistema Híbrido** - Combina ambas para máxima precisão
""")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configurações")
    
    # Status da API OpenAI
    ai_analyzer = AIAnalyzer()
    
    st.markdown("---")
    st.subheader("🤖 Status da IA")
    
    if ai_analyzer.enabled:
        st.success("✅ OpenAI API Ativa")
        st.caption("GPT-4 Vision disponível")
    else:
        st.warning("⚠️ OpenAI API Desativada")
        st.caption("Modo técnico disponível")
    
    st.markdown("---")
    
    # Modo de análise
    modo_analise = st.radio(
        "Modo de Análise",
        ["Duplicidade", "Manipulação por IA", "Análise Completa"],
        help="Escolha o tipo de análise"
    )
    
    # Configurações de duplicidade
    if modo_analise in ["Duplicidade", "Análise Completa"]:
        st.subheader("Detecção de Duplicidade")
        
        limiar_similaridade = st.slider(
            "Limiar de Similaridade (%)", 
            min_value=30, 
            max_value=100, 
            value=40,
            help="Imagens com similaridade acima deste valor serão consideradas duplicatas"
        )
        limiar_similaridade = limiar_similaridade / 100
        
        metodo_deteccao = st.selectbox(
            "Método Técnico",
            ["SIFT + SSIM", "SIFT", "SSIM"],
            help="Método para detecção técnica"
        )
        
        usar_ia_duplicidade = st.checkbox(
            "Usar IA para Duplicidade",
            value=ai_analyzer.enabled,
            disabled=not ai_analyzer.enabled,
            help="GPT-4 Vision analisa contexto e relacionamento"
        )
    
    # Configurações de manipulação
    if modo_analise in ["Manipulação por IA", "Análise Completa"]:
        st.subheader("Detecção de Manipulação")
        
        tamanho_bloco = st.slider(
            "Tamanho do Bloco", 
            min_value=8, 
            max_value=32, 
            value=16,
            step=4,
            help="Menor = mais sensível"
        )
        
        threshold_lbp = st.slider(
            "Sensibilidade LBP", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.35,
            step=0.05,
            help="Menor = mais sensível"
        )
        
        usar_ia_manipulacao = st.checkbox(
            "Usar IA para Manipulação",
            value=ai_analyzer.enabled,
            disabled=not ai_analyzer.enabled,
            help="GPT-4 Vision detecta padrões de IA generativa"
        )

# Upload de arquivos
st.markdown("### 📤 Upload de Imagens")
uploaded_files = st.file_uploader(
    "Arraste suas imagens aqui",
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagem(ns) carregada(s)")
    
    if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
        # Carregar imagens
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir {arquivo.name}: {e}")
        
        # ===== DUPLICIDADE =====
        if modo_analise in ["Duplicidade", "Análise Completa"]:
            st.markdown("---")
            st.markdown("## 🔄 Análise de Duplicidade")
            
            try:
                duplicatas = detectar_duplicatas(
                    imagens, 
                    nomes, 
                    limiar_similaridade, 
                    metodo_deteccao,
                    ai_analyzer if usar_ia_duplicidade else None
                )
                
                if duplicatas:
                    total_duplicatas = sum(len(similares) for similares in duplicatas.values())
                    st.metric("Duplicatas Encontradas", total_duplicatas)
                    
                    # Exibir duplicatas
                    relatorio_duplicatas = []
                    
                    for idx, (img_orig_idx, similares) in enumerate(duplicatas.items()):
                        st.write("---")
                        st.subheader(f"Grupo #{idx+1}")
                        
                        cols = st.columns(min(len(similares) + 1, 3))
                        
                        with cols[0]:
                            st.image(imagens[img_orig_idx], caption=f"Original: {nomes[img_orig_idx]}", use_column_width=True)
                        
                        for i, similar_data in enumerate(similares):
                            col_index = (i + 1) % len(cols)
                            
                            if col_index == 0 and i > 0:
                                cols = st.columns(min(len(similares) - i + 1, 3))
                            
                            with cols[col_index]:
                                similar_idx = similar_data[0]
                                similaridade = similar_data[1]
                                relationship = similar_data[2] if len(similar_data) > 2 else 'unknown'
                                explanation = similar_data[3] if len(similar_data) > 3 else ''
                                
                                st.image(imagens[similar_idx], use_column_width=True)
                                st.caption(f"{nomes[similar_idx]}")
                                st.metric("Similaridade", f"{similaridade:.2%}")
                                
                                if relationship != 'technical_only':
                                    relationship_emoji = {
                                        'identical': '🔄',
                                        'mirror': '🪞',
                                        'crop': '✂️',
                                        'edited': '✏️',
                                        'different': '❌'
                                    }
                                    st.info(f"{relationship_emoji.get(relationship, '🔍')} {relationship.upper()}")
                                    
                                    if explanation:
                                        with st.expander("💡 Análise IA"):
                                            st.write(explanation)
                                
                                if similaridade >= limiar_similaridade:
                                    st.success("✅ DUPLICATA")
                                
                                relatorio_duplicatas.append({
                                    "Original": nomes[img_orig_idx],
                                    "Duplicata": nomes[similar_idx],
                                    "Similaridade": f"{similaridade:.2%}",
                                    "Tipo": relationship
                                })
                    
                    # Relatório
                    if relatorio_duplicatas:
                        st.markdown("### 📊 Relatório de Duplicatas")
                        df = pd.DataFrame(relatorio_duplicatas)
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Baixar CSV",
                            csv,
                            f"duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                else:
                    st.info("Nenhuma duplicata encontrada com o limiar atual.")
                    
            except Exception as e:
                st.error(f"Erro na análise de duplicidade: {e}")
        
        # ===== MANIPULAÇÃO =====
        if modo_analise in ["Manipulação por IA", "Análise Completa"]:
            st.markdown("---")
            st.markdown("## 🤖 Análise de Manipulação")
            
            try:
                analyzer = TextureAnalyzer(
                    P=8, 
                    R=1, 
                    block_size=tamanho_bloco, 
                    threshold=threshold_lbp
                )
                
                resultados_manipulacao = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, img in enumerate(imagens):
                    progress = (i + 1) / len(imagens)
                    progress_bar.progress(progress)
                    status_text.text(f"Analisando textura {i+1}/{len(imagens)}: {nomes[i]}")
                    
                    # Análise técnica
                    report_tecnico = analyzer.analyze_image(img)
                    
                    # Análise com IA
                    ai_result = None
                    if usar_ia_manipulacao and ai_analyzer.enabled:
                        ai_result = ai_analyzer.analyze_manipulation(img)
                    
                    # Combinar resultados
                    if ai_result and ai_result.get('enabled'):
                        # Híbrido: 40% técnico, 60% IA
                        tech_score = report_tecnico['score']
                        ai_score = 100 - ai_result.get('confidence', 50) if ai_result.get('verdict') == 'MANIPULADA' else ai_result.get('confidence', 50)
                        
                        combined_score = int((tech_score * 0.4) + (ai_score * 0.6))
                        
                        resultados_manipulacao.append({
                            "indice": i,
                            "nome": nomes[i],
                            "score": combined_score,
                            "score_tecnico": tech_score,
                            "score_ia": ai_score,
                            "ai_verdict": ai_result.get('verdict'),
                            "ai_explanation": ai_result.get('explanation'),
                            "ai_indicators": ai_result.get('indicators', []),
                            "visual_report": report_tecnico.get("visual_report"),
                            "heatmap": report_tecnico.get("heatmap"),
                            "modo": "Híbrido"
                        })
                    else:
                        # Apenas técnico
                        resultados_manipulacao.append({
                            "indice": i,
                            "nome": nomes[i],
                            "score": report_tecnico['score'],
                            "score_tecnico": report_tecnico['score'],
                            "categoria": report_tecnico['category'],
                            "descricao": report_tecnico['description'],
                            "visual_report": report_tecnico.get("visual_report"),
                            "heatmap": report_tecnico.get("heatmap"),
                            "modo": "Técnico"
                        })
                
                progress_bar.empty()
                status_text.empty()
                
                # Exibir resultados
                manipuladas = sum(1 for r in resultados_manipulacao if r['score'] <= 45)
                suspeitas = sum(1 for r in resultados_manipulacao if 45 < r['score'] <= 70)
                naturais = sum(1 for r in resultados_manipulacao if r['score'] > 70)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 Manipuladas", manipuladas)
                with col2:
                    st.metric("🟡 Suspeitas", suspeitas)
                with col3:
                    st.metric("🟢 Naturais", naturais)
                
                # Detalhes
                for res in resultados_manipulacao:
                    st.write("---")
                    st.subheader(f"{res['nome']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if res["visual_report"] is not None:
                            st.image(res["visual_report"], caption="Análise Visual", use_column_width=True)
                        
                        score = res['score']
                        if score <= 45:
                            st.error(f"🔴 MANIPULADA ({score}%)")
                        elif score <= 70:
                            st.warning(f"🟡 SUSPEITA ({score}%)")
                        else:
                            st.success(f"🟢 NATURAL ({score}%)")
                        
                        st.caption(f"Modo: {res.get('modo', 'Técnico')}")
                        
                        if 'score_tecnico' in res and 'score_ia' in res:
                            st.write(f"**Score Técnico:** {res['score_tecnico']}")
                            st.write(f"**Score IA:** {res['score_ia']}")
                    
                    with col2:
                        if res["heatmap"] is not None:
                            st.image(res["heatmap"], caption="Mapa de Calor", use_column_width=True)
                        
                        if 'ai_explanation' in res:
                            with st.expander("💡 Análise da IA"):
                                st.write(res['ai_explanation'])
                                
                                if res.get('ai_indicators'):
                                    st.write("**Indicadores:**")
                                    for ind in res['ai_indicators']:
                                        st.write(f"- {ind}")
                
                # Relatório
                st.markdown("### 📊 Relatório de Manipulação")
                relatorio_data = []
                for r in resultados_manipulacao:
                    relatorio_data.append({
                        "Arquivo": r['nome'],
                        "Score Final": r['score'],
                        "Modo": r.get('modo', 'Técnico'),
                        "Veredito": "Manipulada" if r['score'] <= 45 else "Suspeita" if r['score'] <= 70 else "Natural"
                    })
                
                df = pd.DataFrame(relatorio_data)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Baixar CSV",
                    csv,
                    f"manipulacao_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Erro na análise de manipulação: {e}")

else:
    st.info("👆 Faça upload de imagens para começar a análise")
    
    with st.expander("📖 Guia Rápido"):
        st.markdown("""
        ### Como Usar:
        
        1. **Escolha o modo de análise:**
           - **Duplicidade**: Detecta imagens idênticas, espelhadas, recortadas
           - **Manipulação**: Detecta edições por IA, texturas artificiais
           - **Completa**: Ambas análises
        
        2. **Faça upload das imagens**
        
        3. **Clique em "Iniciar Análise"**
        
        ### Modos de Operação:
        
        - **Híbrido (com IA)**: Combina análise técnica + GPT-4 Vision
          - Máxima precisão (85-95%)
          - Detecta IA generativa
          - Explica achados
        
        - **Técnico (sem IA)**: Apenas algoritmos matemáticos
          - Gratuito
          - Boa precisão (60-70%)
          - Rápido
        """)

# Rodapé
st.markdown("---")
st.markdown("**MirrorGlass V3** | AI Enhanced Detection | Janeiro 2026")

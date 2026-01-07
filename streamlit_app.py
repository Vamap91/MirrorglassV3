import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pickle
import os
from pathlib import Path
import json
import time
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import euclidean

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mirrorglass Versão 3 - Análise de Duplicidade",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("📊 Mirrorglass Versão 3 - Análise de Duplicidade")
st.markdown("""
Sistema inteligente de detecção de fraudes em imagens usando múltiplos algoritmos.
Detecta imagens duplicadas, recortadas, rotacionadas, espelhadas ou com pequenas alterações.
""")

# Diretórios para armazenar dados
DATABASE_DIR = "image_database"
VECTORS_FILE = os.path.join(DATABASE_DIR, "vectors.pkl")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")

# Criar diretório se não existir
Path(DATABASE_DIR).mkdir(exist_ok=True)

class AdvancedImageVectorizer:
    """Classe para vetorizar imagens usando múltiplos algoritmos robustos"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
    
    def extract_sift_features(self, gray):
        """Extrai features SIFT"""
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
    
    def extract_orb_features(self, gray):
        """Extrai features ORB (mais robusto para rotações)"""
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des
    
    def extract_hu_moments(self, gray):
        """Calcula momentos de Hu (invariante a rotação, escala e translação)"""
        try:
            # Threshold para binarizar
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return np.zeros(7)
            
            # Pegar maior contorno
            cnt = max(contours, key=cv2.contourArea)
            
            # Calcular momentos de Hu
            moments = cv2.HuMoments(cnt)
            return moments.flatten()
        except:
            return np.zeros(7)
    
    def extract_color_histogram(self, image):
        """Calcula histograma de cores (independente de rotação)"""
        if len(image.shape) == 3:
            hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
            hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        else:
            hist = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
        
        hist = hist / (hist.sum() + 1e-7)
        return hist
    
    def extract_edge_features(self, gray):
        """Extrai características de bordas usando Canny"""
        edges = cv2.Canny(gray, 100, 200)
        
        # Calcular características das bordas
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Dividir em grid e calcular densidade por região
        h, w = edges.shape
        grid_size = 4
        grid_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1 = (i * h) // grid_size
                y2 = ((i + 1) * h) // grid_size
                x1 = (j * w) // grid_size
                x2 = ((j + 1) * w) // grid_size
                
                region = edges[y1:y2, x1:x2]
                region_density = np.sum(region > 0) / (region.shape[0] * region.shape[1] + 1e-7)
                grid_features.append(region_density)
        
        return np.array([edge_density] + grid_features)
    
    def extract_texture_features(self, gray):
        """Extrai características de textura usando LBP simplificado"""
        # Calcular variância local
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Erosão e dilatação para análise de textura
        eroded = cv2.erode(gray, kernel, iterations=1)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Diferença entre dilatação e erosão
        texture = dilated - eroded
        
        # Calcular estatísticas
        texture_mean = np.mean(texture)
        texture_std = np.std(texture)
        texture_energy = np.sum(texture ** 2) / (texture.shape[0] * texture.shape[1])
        
        return np.array([texture_mean, texture_std, texture_energy])
    
    def compare_features(self, des1, des2, kp1, kp2):
        """Compara descritores SIFT/ORB"""
        if des1 is None or des2 is None:
            return 0.0
        
        if len(des1) < 2 or len(des2) < 2:
            return 0.0
        
        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            max_matches = min(len(kp1), len(kp2))
            if max_matches == 0:
                return 0.0
            
            similarity = len(good_matches) / max_matches
            return min(1.0, similarity * 2)
        
        except:
            return 0.0
    
    def compare_hu_moments(self, hu1, hu2):
        """Compara momentos de Hu usando distância euclidiana"""
        # Usar log para normalizar
        hu1_log = np.log(np.abs(hu1) + 1e-10)
        hu2_log = np.log(np.abs(hu2) + 1e-10)
        
        # Calcular distância euclidiana normalizada
        distance = euclidean(hu1_log, hu2_log)
        
        # Converter para similaridade (0-1)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def compare_histograms(self, hist1, hist2):
        """Compara histogramas"""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def compare_edge_features(self, edges1, edges2):
        """Compara características de bordas"""
        distance = euclidean(edges1, edges2)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def compare_texture_features(self, texture1, texture2):
        """Compara características de textura"""
        distance = euclidean(texture1, texture2)
        similarity = 1.0 / (1.0 + distance)
        return similarity

class ImageDatabase:
    """Gerencia banco de dados de imagens vetorizadas"""
    
    def __init__(self):
        self.vectorizer = AdvancedImageVectorizer()
        self.database = {}
        self.metadata = {}
        self.load_database()
    
    def load_database(self):
        """Carrega banco de dados do disco"""
        if os.path.exists(VECTORS_FILE):
            try:
                with open(VECTORS_FILE, 'rb') as f:
                    self.database = pickle.load(f)
            except:
                self.database = {}
        
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
    
    def save_database(self):
        """Salva banco de dados no disco"""
        with open(VECTORS_FILE, 'wb') as f:
            pickle.dump(self.database, f)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_image(self, image, image_name):
        """Adiciona imagem ao banco de dados com múltiplos vetores"""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Extrair todas as features
        kp_sift, des_sift = self.vectorizer.extract_sift_features(gray)
        kp_orb, des_orb = self.vectorizer.extract_orb_features(gray)
        hu_moments = self.vectorizer.extract_hu_moments(gray)
        color_hist = self.vectorizer.extract_color_histogram(image)
        edge_features = self.vectorizer.extract_edge_features(gray)
        texture_features = self.vectorizer.extract_texture_features(gray)
        
        image_id = image_name.replace('.', '_').replace(' ', '_')
        
        self.database[image_id] = {
            'name': image_name,
            'kp_sift': len(kp_sift) if kp_sift else 0,
            'des_sift': des_sift,
            'kp_orb': len(kp_orb) if kp_orb else 0,
            'des_orb': des_orb,
            'hu_moments': hu_moments,
            'color_hist': color_hist,
            'edge_features': edge_features,
            'texture_features': texture_features,
            'added_at': datetime.now().isoformat()
        }
        
        self.metadata[image_id] = {
            'name': image_name,
            'added_at': datetime.now().isoformat(),
            'num_features_sift': len(kp_sift) if kp_sift else 0,
            'num_features_orb': len(kp_orb) if kp_orb else 0
        }
        
        self.save_database()
        return image_id
    
    def search_similar(self, image, threshold=0.5):
        """Busca imagens similares usando múltiplos algoritmos"""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Extrair features da imagem de busca
        kp_sift_q, des_sift_q = self.vectorizer.extract_sift_features(gray)
        kp_orb_q, des_orb_q = self.vectorizer.extract_orb_features(gray)
        hu_moments_q = self.vectorizer.extract_hu_moments(gray)
        color_hist_q = self.vectorizer.extract_color_histogram(image)
        edge_features_q = self.vectorizer.extract_edge_features(gray)
        texture_features_q = self.vectorizer.extract_texture_features(gray)
        
        if des_sift_q is None and des_orb_q is None:
            return []
        
        results = []
        
        for image_id, data in self.database.items():
            scores = []
            weights = []
            
            # 1. Comparar SIFT (peso: 0.25)
            if des_sift_q is not None and data['des_sift'] is not None:
                sift_score = self.vectorizer.compare_features(
                    des_sift_q, data['des_sift'],
                    kp_sift_q if kp_sift_q else [],
                    [None] * data['kp_sift']
                )
                scores.append(sift_score)
                weights.append(0.25)
            
            # 2. Comparar ORB (peso: 0.25) - Melhor para rotações
            if des_orb_q is not None and data['des_orb'] is not None:
                orb_score = self.vectorizer.compare_features(
                    des_orb_q, data['des_orb'],
                    kp_orb_q if kp_orb_q else [],
                    [None] * data['kp_orb']
                )
                scores.append(orb_score)
                weights.append(0.25)
            
            # 3. Comparar Momentos de Hu (peso: 0.20) - Invariante a rotação
            hu_score = self.vectorizer.compare_hu_moments(hu_moments_q, data['hu_moments'])
            scores.append(hu_score)
            weights.append(0.20)
            
            # 4. Comparar Histogramas de Cor (peso: 0.15)
            hist_score = self.vectorizer.compare_histograms(color_hist_q, data['color_hist'])
            scores.append(hist_score)
            weights.append(0.15)
            
            # 5. Comparar Bordas (peso: 0.08)
            edge_score = self.vectorizer.compare_edge_features(edge_features_q, data['edge_features'])
            scores.append(edge_score)
            weights.append(0.08)
            
            # 6. Comparar Textura (peso: 0.07)
            texture_score = self.vectorizer.compare_texture_features(texture_features_q, data['texture_features'])
            scores.append(texture_score)
            weights.append(0.07)
            
            # Calcular score ponderado
            total_weight = sum(weights)
            combined_score = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0
            
            if combined_score >= threshold:
                results.append({
                    'image_id': image_id,
                    'name': data['name'],
                    'sift_score': float(scores[0]) if len(scores) > 0 else 0,
                    'orb_score': float(scores[1]) if len(scores) > 1 else 0,
                    'hu_score': float(scores[2]) if len(scores) > 2 else 0,
                    'histogram_score': float(scores[3]) if len(scores) > 3 else 0,
                    'edge_score': float(scores[4]) if len(scores) > 4 else 0,
                    'texture_score': float(scores[5]) if len(scores) > 5 else 0,
                    'combined_score': float(combined_score),
                    'added_at': data['added_at']
                })
        
        # Ordenar por score decrescente
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results

# Inicializar banco de dados
db = ImageDatabase()

# Sidebar
st.sidebar.header("⚙️ Configurações")

# Seleção de modo
mode = st.sidebar.radio(
    "Modo de Operação",
    ["Buscar Duplicatas", "Gerenciar Banco de Dados"],
    help="Escolha o modo de operação"
)

# Limiar de similaridade
threshold = st.sidebar.slider(
    "Limiar de Similaridade",
    min_value=0.3,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Quanto maior, mais rigoroso na detecção"
)

if mode == "Buscar Duplicatas":
    st.markdown("### 🔍 Buscar Imagens Duplicadas")
    
    st.write(f"**Banco de dados contém: {len(db.database)} imagens**")
    
    if len(db.database) == 0:
        st.warning("⚠️ Banco de dados vazio! Adicione imagens de referência primeiro.")
    else:
        # Upload de imagem para busca
        uploaded_file = st.file_uploader(
            "Faça upload de uma imagem para buscar duplicatas",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            # Exibir imagem
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagem de Busca", use_column_width=True)
            
            # Buscar duplicatas
            if st.button("🔎 Buscar Duplicatas", key="search_button"):
                with st.spinner("Analisando imagem com múltiplos algoritmos..."):
                    results = db.search_similar(image, threshold=threshold)
                
                with col2:
                    if results:
                        st.success(f"✅ Encontradas {len(results)} imagem(ns) similar(es)!")
                        
                        for idx, result in enumerate(results, 1):
                            st.markdown(f"### Resultado #{idx}")
                            st.write(f"**Nome:** {result['name']}")
                            st.write(f"**Score Combinado:** {result['combined_score']:.2%}")
                            
                            with st.expander("📊 Detalhes dos Algoritmos"):
                                st.write(f"  - SIFT: {result['sift_score']:.2%}")
                                st.write(f"  - ORB (Rotações): {result['orb_score']:.2%}")
                                st.write(f"  - Momentos de Hu (Invariante): {result['hu_score']:.2%}")
                                st.write(f"  - Histograma de Cores: {result['histogram_score']:.2%}")
                                st.write(f"  - Análise de Bordas: {result['edge_score']:.2%}")
                                st.write(f"  - Análise de Textura: {result['texture_score']:.2%}")
                            
                            st.write(f"**Adicionado em:** {result['added_at']}")
                            
                            if result['combined_score'] >= 0.75:
                                st.error("🚨 ALTA PROBABILIDADE DE DUPLICATA")
                            elif result['combined_score'] >= 0.60:
                                st.warning("⚠️ POSSÍVEL DUPLICATA")
                            else:
                                st.info("ℹ️ Similaridade Detectada")
                            
                            st.divider()
                    else:
                        st.info("ℹ️ Nenhuma imagem similar encontrada no banco de dados.")

elif mode == "Gerenciar Banco de Dados":
    st.markdown("### 📦 Gerenciar Banco de Dados de Imagens")
    
    tab1, tab2, tab3 = st.tabs(["Adicionar Imagens", "Visualizar Banco", "Limpar Banco"])
    
    with tab1:
        st.subheader("Adicionar Novas Imagens de Referência")
        
        uploaded_files = st.file_uploader(
            "Faça upload de imagens para o banco de dados",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("✅ Adicionar ao Banco de Dados"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processando {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    image = Image.open(uploaded_file).convert('RGB')
                    image_id = db.add_image(image, uploaded_file.name)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ {len(uploaded_files)} imagem(ns) adicionada(s) ao banco de dados!")
                time.sleep(1)
                try:
                    st.rerun()
                except:
                    st.experimental_rerun()
    
    with tab2:
        st.subheader("Imagens no Banco de Dados")
        
        if len(db.database) == 0:
            st.info("Banco de dados vazio")
        else:
            # Criar tabela com informações
            data_list = []
            for image_id, data in db.database.items():
                data_list.append({
                    'Nome': data['name'],
                    'Features SIFT': data['kp_sift'],
                    'Features ORB': data['kp_orb'],
                    'Adicionado em': data['added_at'][:10]
                })
            
            st.dataframe(data_list, use_container_width=True)
            
            st.metric("Total de Imagens", len(db.database))
    
    with tab3:
        st.subheader("Limpar Banco de Dados")
        
        if st.button("🗑️ Limpar Banco de Dados", key="clear_db"):
            if st.checkbox("Confirmar exclusão de todas as imagens"):
                db.database = {}
                db.metadata = {}
                db.save_database()
                st.success("✅ Banco de dados limpo!")
                time.sleep(1)
                try:
                    st.rerun()
                except:
                    st.experimental_rerun()

# Rodapé
st.markdown("---")
st.markdown("""
### 📊 Algoritmos Utilizados

**1. SIFT (Scale-Invariant Feature Transform)** - 25%
- Detecta características invariantes à escala
- Bom para recortes e mudanças de tamanho

**2. ORB (Oriented FAST and Rotated BRIEF)** - 25%
- Robusto para rotações e transformações
- Excelente para imagens viradas

**3. Momentos de Hu** - 20%
- Invariante a rotação, escala e translação
- Detecta mesma estrutura em qualquer orientação

**4. Histograma de Cores** - 15%
- Independente de posição e orientação
- Detecta mesma composição de cores

**5. Análise de Bordas** - 8%
- Detecta estrutura de contornos
- Robusto a pequenas edições

**6. Análise de Textura** - 7%
- Detecta padrões de textura
- Invariante a rotações simples

### Limiar de Similaridade

- **0.3 - 0.5**: Muito sensível (pode ter falsos positivos)
- **0.5 - 0.7**: Equilibrado (recomendado)
- **0.7 - 1.0**: Rigoroso (apenas duplicatas óbvias)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
### 📊 Mirrorglass Versão 3
Sistema de Detecção de Fraudes em Imagens

Desenvolvido com Streamlit, OpenCV e Visão Computacional Avançada
""")

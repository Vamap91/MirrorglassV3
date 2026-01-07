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
Sistema inteligente de detecção de fraudes em imagens usando vetorização e análise de características.
Detecta imagens duplicadas, recortadas, rotacionadas ou com pequenas alterações.
""")

# Diretórios para armazenar dados
DATABASE_DIR = "image_database"
VECTORS_FILE = os.path.join(DATABASE_DIR, "vectors.pkl")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")

# Criar diretório se não existir
Path(DATABASE_DIR).mkdir(exist_ok=True)

class ImageVectorizer:
    """Classe para vetorizar imagens usando SIFT e descritores"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
    
    def extract_features(self, image):
        """Extrai features SIFT de uma imagem"""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des, gray
    
    def calculate_histogram(self, image):
        """Calcula histograma de cores como feature adicional"""
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        hist = hist / (hist.sum() + 1e-7)
        
        return hist
    
    def compare_images(self, des1, des2, kp1, kp2):
        """Compara descritores entre duas imagens"""
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
        
        except Exception as e:
            return 0.0
    
    def compare_histograms(self, hist1, hist2):
        """Compara histogramas usando correlação"""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

class ImageDatabase:
    """Gerencia banco de dados de imagens vetorizadas"""
    
    def __init__(self):
        self.vectorizer = ImageVectorizer()
        self.database = {}
        self.metadata = {}
        self.load_database()
    
    def load_database(self):
        """Carrega banco de dados do disco"""
        if os.path.exists(VECTORS_FILE):
            with open(VECTORS_FILE, 'rb') as f:
                self.database = pickle.load(f)
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
    
    def save_database(self):
        """Salva banco de dados no disco"""
        with open(VECTORS_FILE, 'wb') as f:
            pickle.dump(self.database, f)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_image(self, image, image_name):
        """Adiciona imagem ao banco de dados"""
        kp, des, gray = self.vectorizer.extract_features(image)
        hist = self.vectorizer.calculate_histogram(image)
        
        image_id = image_name.replace('.', '_')
        
        self.database[image_id] = {
            'keypoints': len(kp) if kp else 0,
            'descriptors': des,
            'histogram': hist,
            'name': image_name,
            'added_at': datetime.now().isoformat()
        }
        
        self.metadata[image_id] = {
            'name': image_name,
            'added_at': datetime.now().isoformat(),
            'num_features': len(kp) if kp else 0
        }
        
        self.save_database()
        return image_id
    
    def search_similar(self, image, threshold=0.5):
        """Busca imagens similares no banco de dados"""
        kp_query, des_query, gray_query = self.vectorizer.extract_features(image)
        hist_query = self.vectorizer.calculate_histogram(image)
        
        if des_query is None:
            return []
        
        results = []
        
        for image_id, data in self.database.items():
            if data['descriptors'] is None:
                continue
            
            # Comparar descritores SIFT
            sift_score = self.vectorizer.compare_images(
                des_query, 
                data['descriptors'],
                kp_query if kp_query else [],
                [None] * data['keypoints']
            )
            
            # Comparar histogramas
            hist_score = self.vectorizer.compare_histograms(
                hist_query,
                data['histogram']
            )
            
            # Score combinado (SIFT tem mais peso)
            combined_score = (sift_score * 0.7) + (hist_score * 0.3)
            
            if combined_score >= threshold:
                results.append({
                    'image_id': image_id,
                    'name': data['name'],
                    'sift_score': float(sift_score),
                    'histogram_score': float(hist_score),
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
    value=0.5,
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
                with st.spinner("Analisando imagem..."):
                    results = db.search_similar(image, threshold=threshold)
                
                with col2:
                    if results:
                        st.success(f"✅ Encontradas {len(results)} imagem(ns) similar(es)!")
                        
                        for idx, result in enumerate(results, 1):
                            st.markdown(f"### Resultado #{idx}")
                            st.write(f"**Nome:** {result['name']}")
                            st.write(f"**Score Combinado:** {result['combined_score']:.2%}")
                            st.write(f"  - SIFT: {result['sift_score']:.2%}")
                            st.write(f"  - Histograma: {result['histogram_score']:.2%}")
                            st.write(f"**Adicionado em:** {result['added_at']}")
                            
                            if result['combined_score'] >= 0.7:
                                st.error("🚨 ALTA PROBABILIDADE DE DUPLICATA")
                            elif result['combined_score'] >= 0.5:
                                st.warning("⚠️ POSSÍVEL DUPLICATA")
                            
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
                st.rerun()
    
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
                    'Features': data['keypoints'],
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
                st.rerun()

# Rodapé
st.markdown("---")
st.markdown("""
### 📊 Como funciona

**Sistema de Vetorização:**
- Extrai características SIFT de cada imagem
- Calcula histogramas de cores
- Armazena vetores em banco de dados persistente

**Detecção de Duplicatas:**
- Compara features SIFT (70% do score)
- Compara histogramas de cores (30% do score)
- Detecta imagens recortadas, rotacionadas ou alteradas

**Limiar de Similaridade:**
- 0.3 - 0.5: Muito sensível (pode ter falsos positivos)
- 0.5 - 0.7: Equilibrado (recomendado)
- 0.7 - 1.0: Rigoroso (apenas duplicatas óbvias)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
### 📊 Mirrorglass Versão 3
Sistema de Detecção de Fraudes em Imagens

Desenvolvido com Streamlit, OpenCV e SIFT
""")

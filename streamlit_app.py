import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import json
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import pandas as pd
import time
import cv2

# Configuração da página Streamlit
st.set_page_config(
    page_title="Mirrorglass Versão 3 - Análise de Duplicidade",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 Mirrorglass Versão 3 - Análise de Duplicidade")
st.markdown("""
Este sistema utiliza técnicas avançadas de visão computacional para:
- **Detectar imagens duplicadas** ou altamente semelhantes, mesmo com alterações como cortes ou ajustes

### Como funciona?
1. Faça upload das imagens para análise
2. O sistema analisa duplicidade usando SIFT e SSIM
3. Resultados são exibidos com detalhamento visual e score de similaridade
""")

# Barra lateral com controles
st.sidebar.header("⚙️ Configurações")

# Configurações para detecção de duplicidade
st.sidebar.subheader("Configurações de Duplicidade")
limiar_similaridade = st.sidebar.slider(
    "Limiar de Similaridade (%)", 
    min_value=30, 
    max_value=100, 
    value=50, 
    help="Imagens com similaridade acima deste valor serão consideradas possíveis duplicatas"
)
limiar_similaridade = limiar_similaridade / 100  # Converter para decimal

metodo_deteccao = st.sidebar.selectbox(
    "Método de Detecção",
    ["SIFT (melhor para recortes)", "SSIM + SIFT", "SSIM"],
    help="Escolha o método para detectar imagens similares"
)

# Funções para processamento de imagens - DUPLICIDADE
def preprocessar_imagem(img, tamanho=(300, 300)):
    try:
        # Redimensionar
        img_resize = img.resize(tamanho)
        # Converter para escala de cinza para SSIM
        img_gray = img_resize.convert('L')
        # Converter para array numpy
        img_array = np.array(img_gray)
        # Normalizar valores para [0, 1]
        img_array = img_array / 255.0
        # Converter para CV2 formato (para SIFT)
        img_cv = np.array(img_resize)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB para BGR
        return img_array, img_cv
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None

def calcular_similaridade_ssim(img1, img2):
    try:
        # Garantir que as imagens tenham o mesmo tamanho
        if img1.shape != img2.shape:
            img2 = resize(img2, img1.shape)
        
        # Calcular SSIM com data_range especificado
        score = ssim(img1, img2, data_range=1.0)
        return score
    except Exception as e:
        st.error(f"Erro ao calcular similaridade SSIM: {e}")
        return 0

def calcular_similaridade_sift(img1_cv, img2_cv):
    try:
        # Converter para escala de cinza
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        
        # Inicializar o detector SIFT
        sift = cv2.SIFT_create()
        
        # Detectar keypoints e descritores
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        # Se não houver descritores suficientes, retorna 0
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
            
        # Usar o matcher FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Encontrar os 2 melhores matches para cada descritor
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filtrar bons matches usando o teste de proporção de Lowe
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Calcular a similaridade baseada no número de bons matches
        max_matches = min(len(kp1), len(kp2))
        if max_matches == 0:
            return 0
            
        similarity = len(good_matches) / max_matches
        
        # Normalizar para evitar valores muito baixos
        if similarity < 0.05:
            adjusted_similarity = 0
        else:
            # Expandir valores pequenos para uma escala mais ampla
            adjusted_similarity = min(1.0, similarity * 2)
        
        return adjusted_similarity
        
    except Exception as e:
        st.error(f"Erro ao calcular similaridade SIFT: {e}")
        return 0

def calcular_similaridade_combinada(img1_gray, img2_gray, img1_cv, img2_cv):
    try:
        # Calcular similaridade usando ambos os métodos
        sim_ssim = calcular_similaridade_ssim(img1_gray, img2_gray)
        sim_sift = calcular_similaridade_sift(img1_cv, img2_cv)
        
        # A similaridade combinada é a média ponderada dos dois valores
        # SIFT tem mais peso para detectar recortes
        return (sim_ssim * 0.3) + (sim_sift * 0.7)
    except Exception as e:
        st.error(f"Erro ao calcular similaridade combinada: {e}")
        return 0

def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_image_download_link(img, filename, text):
    # Converter para PIL Image se for numpy array
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
        
    # Salvar em buffer
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)
    
    # Codificar para base64
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    
    return href

def visualizar_duplicatas(imagens, nomes, duplicatas, limiar):
    if not duplicatas:
        st.info("Nenhuma duplicata encontrada com o limiar de similaridade atual.")
        return None
    
    # Criar DataFrame para relatório
    relatorio_dados = []
    
    # Para cada grupo de duplicatas
    for idx, (img_orig_idx, similares) in enumerate(duplicatas.items()):
        st.write("---")
        st.subheader(f"Grupo de Duplicatas #{idx+1}")
        
        # Layout para imagem original e suas duplicatas
        cols = st.columns(min(len(similares) + 1, 4))  # Limita a 4 colunas por linha
        
        # Mostrar imagem original
        with cols[0]:
            st.image(imagens[img_orig_idx], caption=f"Original: {nomes[img_orig_idx]}", width=200)
        
        # Mostrar duplicatas
        for i, (similar_idx, similaridade) in enumerate(similares):
            col_index = (i + 1) % len(cols)
            
            # Se precisar de uma nova linha
            if col_index == 0 and i > 0:
                st.write("")  # Linha em branco
                cols = st.columns(min(len(similares) - i + 1, 4))
            
            with cols[col_index]:
                st.image(imagens[similar_idx], width=200)
                caption = f"{nomes[similar_idx]}\nSimilaridade: {similaridade:.2f}"
                st.caption(caption)
                
                # Destacar em verde se acima do limiar
                if similaridade >= limiar:
                    st.success("DUPLICATA DETECTADA")
                
                # Adicionar ao relatório
                relatorio_dados.append({
                    "Arquivo Original": nomes[img_orig_idx],
                    "Arquivo Duplicado": nomes[similar_idx],
                    "Similaridade (%)": round(similaridade * 100, 2)
                })
    
    # Criar DataFrame do relatório
    if relatorio_dados:
        df_relatorio = pd.DataFrame(relatorio_dados)
        return df_relatorio
    return None

# Função principal para detectar duplicatas
def detectar_duplicatas(imagens, nomes, limiar=0.5, metodo="SIFT (melhor para recortes)"):
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processar imagens
    status_text.text("Extraindo características das imagens...")
    arrays_processados_gray = []  # Para SSIM
    arrays_processados_cv = []    # Para SIFT
    indices_validos = []
    
    for i, img in enumerate(imagens):
        # Atualizar barra de progresso
        progress = (i + 1) / len(imagens)
        progress_bar.progress(progress)
        status_text.text(f"Processando imagem {i+1} de {len(imagens)}: {nomes[i]}")
        
        # Preprocessar imagem
        img_array_gray, img_array_cv = preprocessar_imagem(img)
        if img_array_gray is not None:
            arrays_processados_gray.append(img_array_gray)
            arrays_processados_cv.append(img_array_cv)
            indices_validos.append(i)
    
    if not arrays_processados_gray:
        status_text.error("Nenhuma imagem válida para processamento.")
        progress_bar.empty()
        return None
    
    # Calcular similaridades
    status_text.text("Comparando imagens e buscando duplicatas...")
    duplicatas = {}  # {índice_original: [(índice_similar, similaridade), ...]}
    
    total_comparacoes = len(arrays_processados_gray) * (len(arrays_processados_gray) - 1) // 2
    comparacao_atual = 0
    
    for i in range(len(arrays_processados_gray)):
        similares = []
        for j in range(len(arrays_processados_gray)):
            # Não comparar uma imagem com ela mesma
            if i != j:
                comparacao_atual += 1
                
                # Atualizar progresso de maneira mais segura
                if total_comparacoes > 0:
                    # Certificar que o progresso sempre está entre 0 e 1
                    progress = min(max(comparacao_atual / total_comparacoes, 0.0), 1.0)
                    progress_bar.progress(progress)
                
                # Calcular similaridade com base no método selecionado
                if metodo == "SSIM":
                    similaridade = calcular_similaridade_ssim(
                        arrays_processados_gray[i], 
                        arrays_processados_gray[j]
                    )
                elif metodo == "SIFT (melhor para recortes)":
                    similaridade = calcular_similaridade_sift(
                        arrays_processados_cv[i], 
                        arrays_processados_cv[j]
                    )
                else:  # SSIM + SIFT
                    similaridade = calcular_similaridade_combinada(
                        arrays_processados_gray[i], 
                        arrays_processados_gray[j],
                        arrays_processados_cv[i], 
                        arrays_processados_cv[j]
                    )
                
                # Se acima do limiar, adicionar como duplicata
                if similaridade >= limiar:
                    similares.append((indices_validos[j], similaridade))
        
        # Se encontrou duplicatas, adicionar à lista
        if similares:
            duplicatas[indices_validos[i]] = similares
    
    progress_bar.empty()
    status_text.text("Processamento concluído!")
    
    return duplicatas

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

# Função para gerar JSON resumido
def gerar_json_resumido(dados, tipo_analise):
    if tipo_analise == "Duplicidade":
        if dados:
            resumo = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Duplicidade",
                "metodo_usado": "SSIM + SIFT",
                "total_grupos_duplicatas": len(dados),
                "total_duplicatas_encontradas": sum(len(similares) for similares in dados.values()),
                "resumo_grupos": []
            }
            
            for img_orig_idx, similares in dados.items():
                grupo_resumo = {
                    "imagem_original_indice": img_orig_idx,
                    "quantidade_duplicatas": len(similares),
                    "maior_similaridade": max([sim for _, sim in similares]) if similares else 0,
                    "menor_similaridade": min([sim for _, sim in similares]) if similares else 0
                }
                resumo["resumo_grupos"].append(grupo_resumo)
            
            return resumo
        else:
            return {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "tipo_analise": "Duplicidade",
                "total_grupos_duplicatas": 0,
                "total_duplicatas_encontradas": 0,
                "resultado": "Nenhuma duplicata encontrada"
            }

# Interface principal
st.markdown("### 🔹 Passo 1: Carregar Imagens")
uploaded_files = st.file_uploader(
    "Faça upload das imagens para análise", 
    accept_multiple_files=True,
    type=['jpg', 'jpeg', 'png']
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} imagens carregadas")
    
    # Criar botão para iniciar processamento
    if st.button("🚀 Iniciar Análise de Duplicidade", key="iniciar_analise"):
        # Carregar imagens
        imagens = []
        nomes = []
        
        for arquivo in uploaded_files:
            try:
                img = Image.open(arquivo).convert('RGB')
                imagens.append(img)
                nomes.append(arquivo.name)
            except Exception as e:
                st.error(f"Erro ao abrir a imagem {arquivo.name}: {e}")
        
        # Análise de duplicidade
        try:
            st.markdown("## 🔍 Análise de Duplicidade")
            duplicatas = detectar_duplicatas(imagens, nomes, limiar_similaridade, metodo_deteccao)
            
            # Visualizar resultados de duplicidade
            if duplicatas:
                # Estatísticas
                total_duplicatas = sum(len(similares) for similares in duplicatas.values())
                st.metric("Total de possíveis duplicatas encontradas", total_duplicatas)
                
                # Visualizar duplicatas
                df_relatorio = visualizar_duplicatas(imagens, nomes, duplicatas, limiar_similaridade)
                
                # Gerar relatório
                if df_relatorio is not None:
                    st.markdown("### 🔹 Relatório de Duplicatas")
                    st.dataframe(df_relatorio)
                    
                    # Opção para download do relatório
                    nome_arquivo = f"relatorio_duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    st.markdown(get_csv_download_link(df_relatorio, nome_arquivo, 
                                                 "📥 Baixar Relatório CSV"), unsafe_allow_html=True)
                
                # JSON Resumido
                with st.expander("📄 Ver JSON Resumido - Duplicidade"):
                    json_resumido = gerar_json_resumido(duplicatas, "Duplicidade")
                    json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                    st.code(json_str, language='json')
                    
                    st.download_button(
                        label="📥 Baixar JSON Resumido",
                        data=json_str,
                        file_name=f"resumo_duplicatas_{time.strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Nenhuma duplicata encontrada com o limiar atual. Tente reduzir o limiar de similaridade.")
                
                # JSON vazio para nenhuma duplicata
                with st.expander("📄 Ver JSON Resumido - Duplicidade"):
                    json_resumido = gerar_json_resumido(None, "Duplicidade")
                    json_str = json.dumps(json_resumido, indent=2, ensure_ascii=False)
                    st.code(json_str, language='json')
                    
        except Exception as e:
            st.error(f"Erro durante a detecção de duplicatas: {str(e)}")

else:
    # Mostrar exemplo quando não há imagens carregadas
    st.info("Faça upload de imagens para começar a análise de duplicidade.")
    
    # Adicionar imagens de exemplo
    if st.button("🔍 Ver exemplos de detecção", key="ver_exemplos"):
        st.write("### Exemplo de Detecção de Duplicidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://via.placeholder.com/400x300?text=Original", caption="Imagem Original")
            
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Duplicata+Recortada", caption="Duplicata (Recortada)")
            st.write("Similaridade: 0.78")
            st.success("DUPLICATA DETECTADA")
        
        # Exemplo de JSON Resumido
        with st.expander("📄 Exemplo de JSON Resumido"):
            exemplo_json = {
                "timestamp": "2025-06-06 14:30:00",
                "tipo_analise": "Duplicidade",
                "metodo_usado": "SSIM + SIFT",
                "total_grupos_duplicatas": 1,
                "total_duplicatas_encontradas": 1,
                "resumo_grupos": [
                    {
                        "imagem_original_indice": 0,
                        "quantidade_duplicatas": 1,
                        "maior_similaridade": 0.85,
                        "menor_similaridade": 0.85
                    }
                ]
            }
            
            json_str = json.dumps(exemplo_json, indent=2, ensure_ascii=False)
            st.code(json_str, language='json')

# Rodapé
st.markdown("---")
st.markdown("### Como interpretar os resultados")

# Explicação sobre duplicidade
st.markdown("""
#### 🔍 Análise de Duplicidade

**O que é detectado:**
- Imagens que são cópias exatas ou muito similares
- Imagens recortadas ou com pequenas alterações
- Imagens com diferentes resoluções mas mesmo conteúdo

**Métodos disponíveis:**
- **SIFT**: Melhor para detectar recortes e transformações geométricas
- **SSIM**: Melhor para detectar cópias com pequenas alterações
- **SSIM + SIFT**: Combina ambos os métodos para maior precisão

**Interpretação do score:**
- **0.0 - 0.3**: Imagens diferentes
- **0.3 - 0.6**: Possível duplicata com alterações
- **0.6 - 1.0**: Duplicata provável
""")

# Contato e informações
st.sidebar.markdown("---")
st.sidebar.info("""
### 📊 Mirrorglass Versão 3
Sistema de Análise de Duplicidade em Imagens

Desenvolvido com Streamlit, OpenCV e Scikit-Image
""")

import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Detector de Duplicidade Pro", page_icon="🔄", layout="wide")

def calcular_hash(img):
    img_small = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
    pixels = np.array(img_small).flatten()
    avg = pixels.mean()
    return (pixels > avg).astype(int)

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def comparar_pixel(img1, img2):
    size = 128
    img1_small = np.array(img1.resize((size, size)).convert('RGB'))
    img2_small = np.array(img2.resize((size, size)).convert('RGB'))
    
    diff = np.abs(img1_small.astype(float) - img2_small.astype(float))
    score = 1.0 - (np.mean(diff) / 255.0)
    return score

def comparar_histograma(img1, img2):
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    hist1 = cv2.calcHist([img1_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    
    hist2 = cv2.calcHist([img2_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def comparar_completo(img1, img2):
    hash1 = calcular_hash(img1)
    hash2 = calcular_hash(img2)
    dist = hamming_distance(hash1, hash2)
    hash_sim = 1.0 - (dist / 64.0)
    
    pixel_sim = comparar_pixel(img1, img2)
    hist_sim = comparar_histograma(img1, img2)
    
    score_final = (hash_sim * 0.3 + pixel_sim * 0.4 + hist_sim * 0.3)
    
    return {
        'score': score_final,
        'hash': hash_sim,
        'pixel': pixel_sim,
        'hist': hist_sim
    }

st.title("🔄 Detector de Duplicidade Pro")

with st.sidebar:
    st.header("Configurações")
    limiar = st.slider("Limiar de Duplicata (%)", 50, 95, 75, 5) / 100
    mostrar_detalhes = st.checkbox("Mostrar Detalhes", True)

uploaded = st.file_uploader("📤 Upload Imagens", type=['jpg','png','jpeg'], accept_multiple_files=True)

if uploaded and len(uploaded) >= 2:
    if st.button("🚀 Analisar", type="primary"):
        imgs = []
        nomes = []
        
        for f in uploaded:
            try:
                img = Image.open(f).convert('RGB')
                imgs.append(img)
                nomes.append(f.name)
            except:
                st.error(f"Erro: {f.name}")
        
        if len(imgs) < 2:
            st.error("Precisa de pelo menos 2 imagens válidas")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            duplicatas = []
            total = len(imgs) * (len(imgs) - 1) // 2
            atual = 0
            
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    atual += 1
                    progress.progress(atual / total)
                    status.text(f"Comparando {atual}/{total}")
                    
                    resultado = comparar_completo(imgs[i], imgs[j])
                    
                    if resultado['score'] >= limiar:
                        duplicatas.append({
                            'i': i,
                            'j': j,
                            'nome1': nomes[i],
                            'nome2': nomes[j],
                            'score': resultado['score'],
                            'hash': resultado['hash'],
                            'pixel': resultado['pixel'],
                            'hist': resultado['hist']
                        })
            
            progress.empty()
            status.empty()
            
            if duplicatas:
                st.error(f"🔴 {len(duplicatas)} duplicata(s) encontrada(s)")
                
                for idx, d in enumerate(duplicatas):
                    st.subheader(f"Par #{idx+1}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.image(imgs[d['i']], caption=d['nome1'], use_column_width=True)
                    
                    with col2:
                        st.image(imgs[d['j']], caption=d['nome2'], use_column_width=True)
                    
                    with col3:
                        st.metric("Similaridade", f"{d['score']:.0%}")
                        
                        if mostrar_detalhes:
                            with st.expander("Detalhes"):
                                st.write(f"Hash: {d['hash']:.0%}")
                                st.write(f"Pixel: {d['pixel']:.0%}")
                                st.write(f"Hist: {d['hist']:.0%}")
                    
                    st.markdown("---")
            else:
                st.success("✅ Nenhuma duplicata encontrada")
                st.info(f"Limiar atual: {limiar:.0%}")

elif uploaded and len(uploaded) == 1:
    st.warning("⚠️ Upload pelo menos 2 imagens")
else:
    st.info("👆 Faça upload de 2+ imagens")

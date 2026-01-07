import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Detector de Duplicidade", page_icon="🔄", layout="wide")

def comparar_imagens(img1, img2):
    img1_cv = cv2.cvtColor(np.array(img1.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(np.array(img2.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    
    diff = cv2.absdiff(img1_cv, img2_cv)
    similaridade = 1.0 - (np.mean(diff) / 255.0)
    
    return similaridade

st.title("🔄 Detector de Duplicidade Simples")

limiar = st.sidebar.slider("Limiar (%)", 50, 95, 80) / 100

uploaded = st.file_uploader("Upload Imagens", type=['jpg','png','jpeg'], accept_multiple_files=True)

if uploaded and len(uploaded) >= 2:
    if st.button("Analisar"):
        imgs = [Image.open(f).convert('RGB') for f in uploaded]
        nomes = [f.name for f in uploaded]
        
        duplicatas = []
        
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                sim = comparar_imagens(imgs[i], imgs[j])
                
                if sim >= limiar:
                    duplicatas.append({
                        'i': i,
                        'j': j,
                        'sim': sim,
                        'nome1': nomes[i],
                        'nome2': nomes[j]
                    })
        
        if duplicatas:
            st.error(f"🔴 {len(duplicatas)} duplicata(s)")
            
            for d in duplicatas:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(imgs[d['i']], caption=d['nome1'])
                
                with col2:
                    st.image(imgs[d['j']], caption=d['nome2'])
                
                with col3:
                    st.metric("Similaridade", f"{d['sim']:.0%}")
                
                st.markdown("---")
        else:
            st.success("✅ Nenhuma duplicata")

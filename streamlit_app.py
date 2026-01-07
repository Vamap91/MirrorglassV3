import streamlit as st
import numpy as np
from PIL import Image
import cv2
import base64
import io
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Detector Duplicatas", page_icon="🔍", layout="wide")

if 'base_historica' not in st.session_state:
    st.session_state.base_historica = []
    st.session_state.nomes_historico = []

def image_to_base64(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analisar_contexto_openai(img1, img2, api_key, sift_score):
    openai.api_key = api_key
    
    img1_b64 = image_to_base64(img1)
    img2_b64 = image_to_base64(img2)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""ANÁLISE DE DUPLICATA - Score SIFT: {sift_score:.0%}

Compare estas imagens e responda APENAS em JSON:

{{
    "mesmo_veiculo": true/false,
    "mesmo_local": true/false,
    "mesmo_contexto": true/false,
    "mesma_oficina": true/false,
    "elementos_comuns": ["elemento1", "elemento2"],
    "sao_duplicatas": true/false,
    "confianca": 0-100,
    "motivo": "explicação"
}}

REGRAS:
- Se mesmo veículo + mesma oficina → duplicatas=true
- Se contexto 80%+ idêntico → duplicatas=true
- Se apenas ângulo diferente mas mesma cena → duplicatas=true
- Liste TODOS elementos comuns"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img1_b64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img2_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        return json.loads(result_text)
        
    except Exception as e:
        return {'error': str(e)}

def calcular_sift(img1, img2):
    try:
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0, 0
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        max_matches = min(len(kp1), len(kp2))
        if max_matches == 0:
            return 0, 0
        
        score = len(good) / max_matches
        return score, len(good)
        
    except Exception as e:
        return 0, 0

def detectar_duplicata(img1, img2, api_key, usar_ia):
    sift_score, good_matches = calcular_sift(img1, img2)
    
    if sift_score >= 0.40:
        return {
            'duplicata': True,
            'metodo': 'SIFT',
            'score': sift_score,
            'matches': good_matches,
            'motivo': f'Alta similaridade SIFT: {sift_score:.0%} ({good_matches} matches)'
        }
    
    if sift_score >= 0.15 and usar_ia and api_key:
        ia_result = analisar_contexto_openai(img1, img2, api_key, sift_score)
        
        if 'error' not in ia_result:
            sao_dup = ia_result.get('sao_duplicatas', False)
            mesmo_veiculo = ia_result.get('mesmo_veiculo', False)
            mesmo_contexto = ia_result.get('mesmo_contexto', False)
            confianca = ia_result.get('confianca', 0) / 100.0
            
            if sao_dup or (mesmo_veiculo and mesmo_contexto) or confianca >= 0.70:
                return {
                    'duplicata': True,
                    'metodo': 'IA',
                    'score': max(sift_score, confianca),
                    'matches': good_matches,
                    'motivo': ia_result.get('motivo', 'IA confirmou duplicata'),
                    'ia_detalhes': ia_result
                }
    
    return {
        'duplicata': False,
        'metodo': 'OK',
        'score': sift_score,
        'matches': good_matches,
        'motivo': 'Imagens diferentes'
    }

st.title("🔍 Detector de Duplicatas")
st.markdown("### SIFT + OpenAI Context Analysis")

with st.sidebar:
    st.header("📚 Base")
    
    st.info(f"**Imagens:** {len(st.session_state.base_historica)}")
    
    api_key = None
    usar_ia = False
    
    if OPENAI_AVAILABLE:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                st.success("✅ OpenAI")
                usar_ia = st.checkbox("🤖 Análise IA", value=True)
            else:
                st.error("❌ Configure OPENAI_API_KEY")
        except:
            st.error("❌ Configure OPENAI_API_KEY")
    
    st.markdown("---")
    
    uploaded_base = st.file_uploader(
        "📤 Upload Base",
        type=['jpg','png','jpeg'],
        accept_multiple_files=True,
        key="base"
    )
    
    if uploaded_base:
        if st.button("💾 Adicionar"):
            novos = 0
            for f in uploaded_base:
                if f.name not in st.session_state.nomes_historico:
                    try:
                        img = Image.open(f).convert('RGB')
                        st.session_state.base_historica.append(img)
                        st.session_state.nomes_historico.append(f.name)
                        novos += 1
                    except:
                        pass
            
            if novos > 0:
                st.success(f"✅ +{novos}")
                st.rerun()
    
    if len(st.session_state.base_historica) > 0:
        if st.button("🗑️ Limpar"):
            st.session_state.base_historica = []
            st.session_state.nomes_historico = []
            st.rerun()

st.markdown("---")

if len(st.session_state.base_historica) == 0:
    st.warning("⚠️ Configure base")
else:
    st.success(f"✅ Base: {len(st.session_state.base_historica)}")
    
    st.markdown("---")
    st.header("🆕 Testar Nova")
    
    uploaded_nova = st.file_uploader(
        "📤 Upload Nova",
        type=['jpg','png','jpeg'],
        key="nova"
    )
    
    if uploaded_nova:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🆕 Nova")
            nova_img = Image.open(uploaded_nova).convert('RGB')
            st.image(nova_img, use_column_width=True)
        
        with col2:
            st.subheader(f"📚 Base: {len(st.session_state.base_historica)}")
            if usar_ia:
                st.info("🤖 IA ativa para análise de contexto")
        
        st.markdown("---")
        
        if st.button("🔍 Detectar Duplicatas", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            resultados = []
            
            for idx, img_base in enumerate(st.session_state.base_historica):
                progress.progress((idx + 1) / len(st.session_state.base_historica))
                status.text(f"Analisando {idx + 1}/{len(st.session_state.base_historica)}")
                
                resultado = detectar_duplicata(nova_img, img_base, api_key, usar_ia)
                
                if resultado['duplicata']:
                    resultados.append({
                        'idx': idx,
                        'nome': st.session_state.nomes_historico[idx],
                        'img': img_base,
                        'resultado': resultado
                    })
            
            progress.empty()
            status.empty()
            
            st.markdown("---")
            st.markdown("## 📊 Resultado")
            
            if resultados:
                st.error(f"🚨 {len(resultados)} DUPLICATA(S)")
                
                for r in resultados:
                    st.markdown("---")
                    st.subheader(f"🚨 Duplicata: {r['nome']}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**🆕 Nova**")
                        st.image(nova_img, use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**📚 Base**")
                        st.image(r['img'], caption=r['nome'], use_column_width=True)
                    
                    with col3:
                        res = r['resultado']
                        st.metric("Score", f"{res['score']:.0%}")
                        st.metric("Matches", res['matches'])
                        
                        if res['metodo'] == 'IA':
                            st.error("🤖 IA")
                        else:
                            st.error("🔬 SIFT")
                        
                        st.caption(res['motivo'])
                    
                    if 'ia_detalhes' in res:
                        with st.expander("🔍 Análise IA"):
                            ia = res['ia_detalhes']
                            st.write(f"**Mesmo veículo:** {'✅' if ia.get('mesmo_veiculo') else '❌'}")
                            st.write(f"**Mesmo local:** {'✅' if ia.get('mesmo_local') else '❌'}")
                            st.write(f"**Mesmo contexto:** {'✅' if ia.get('mesmo_contexto') else '❌'}")
                            st.write(f"**Confiança:** {ia.get('confianca')}%")
                            
                            if ia.get('elementos_comuns'):
                                st.write("**Elementos comuns:**")
                                for elem in ia['elementos_comuns']:
                                    st.write(f"• {elem}")
            
            else:
                st.success("✅ Nenhuma duplicata")

st.markdown("---")
st.caption("Detector Duplicatas | SIFT + OpenAI | Janeiro 2026")

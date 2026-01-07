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

st.set_page_config(page_title="Detector Genial Multi-Camadas", page_icon="🧠", layout="wide")

if 'base_historica' not in st.session_state:
    st.session_state.base_historica = []
    st.session_state.nomes_historico = []

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analisar_ia_contexto(img1, img2, api_key, razao_acionamento):
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
                            "text": f"""ANÁLISE CRÍTICA DE DUPLICATA/FRAUDE:

Razão do acionamento: {razao_acionamento}

Compare estas 2 imagens e responda em JSON:

{{
    "descricao_img1": "descrição completa",
    "descricao_img2": "descrição completa",
    "objeto_principal_1": "ex: Carro branco VW",
    "objeto_principal_2": "ex: Carro branco VW",
    "contexto_1": "oficina, jeep ao fundo, parede azul",
    "contexto_2": "oficina, jeep ao fundo, parede azul",
    "mesmo_objeto": true/false,
    "mesmo_contexto": true/false,
    "mesmo_local": true/false,
    "eh_crop_ou_zoom": true/false,
    "eh_mesma_foto": true/false,
    "confianca": 0-100,
    "tipo_duplicata": "EXATA|CROP|EDITADA|DIFERENTES",
    "elementos_identicos": ["elemento1", "elemento2"],
    "explicacao": "explicação detalhada"
}}

ATENÇÃO ESPECIAL:
- Se CONTEXTO idêntico mas ângulo diferente → mesma_foto=true, eh_crop_ou_zoom=true
- Se mesma cena mas uma é CROP/RECORTE da outra → eh_mesma_foto=true
- Se mesmo carro, mesma oficina, mesmos objetos → mesmo_contexto=true
- Se elementos de fundo IDÊNTICOS → provável crop/fraude"""
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
            max_tokens=600
        )
        
        result_text = response.choices[0].message.content
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        return {
            'enabled': True,
            'desc1': result.get('descricao_img1', 'N/A'),
            'desc2': result.get('descricao_img2', 'N/A'),
            'obj1': result.get('objeto_principal_1', 'N/A'),
            'obj2': result.get('objeto_principal_2', 'N/A'),
            'ctx1': result.get('contexto_1', 'N/A'),
            'ctx2': result.get('contexto_2', 'N/A'),
            'mesmo_objeto': result.get('mesmo_objeto', False),
            'mesmo_contexto': result.get('mesmo_contexto', False),
            'mesmo_local': result.get('mesmo_local', False),
            'eh_crop': result.get('eh_crop_ou_zoom', False),
            'eh_mesma_foto': result.get('eh_mesma_foto', False),
            'confianca': result.get('confianca', 0),
            'tipo': result.get('tipo_duplicata', 'N/A'),
            'elementos': result.get('elementos_identicos', []),
            'explicacao': result.get('explicacao', 'N/A')
        }
    except Exception as e:
        return {'enabled': False, 'error': str(e)}

def layer1_duplicata_exata(img1, img2):
    size = 128
    img1_small = np.array(img1.resize((size, size)).convert('RGB'))
    img2_small = np.array(img2.resize((size, size)).convert('RGB'))
    
    diff = np.abs(img1_small.astype(float) - img2_small.astype(float))
    pixel_score = 1.0 - (np.mean(diff) / 255.0)
    
    img1_cv = cv2.cvtColor(img1_small, cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(img2_small, cv2.COLOR_RGB2BGR)
    
    hist1 = cv2.calcHist([img1_cv], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    
    hist2 = cv2.calcHist([img2_cv], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    score = (pixel_score * 0.6) + (hist_score * 0.4)
    
    return {
        'score': score,
        'pixel': pixel_score,
        'hist': hist_score,
        'conclusivo': score >= 0.90
    }

def layer2_crop_rotacao(img1, img2):
    size = 512
    img1_big = np.array(img1.resize((size, size)).convert('RGB'))
    img2_big = np.array(img2.resize((size, size)).convert('RGB'))
    
    gray1 = cv2.cvtColor(img1_big, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_big, cv2.COLOR_RGB2GRAY)
    
    sift = cv2.SIFT_create(nfeatures=500)
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return {
            'matches': 0,
            'good_matches': 0,
            'inliers': 0,
            'score': 0,
            'conclusivo': False,
            'suspeito_crop': False
        }
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    num_good = len(good)
    inliers = 0
    
    if num_good >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None:
                inliers = int(np.sum(mask))
        except:
            pass
    
    score_sift = min(1.0, num_good / 100.0)
    
    suspeito_crop = (num_good >= 30 and inliers >= 15)
    conclusivo = (num_good >= 50 and inliers >= 30 and score_sift >= 0.70)
    
    return {
        'matches': len(matches),
        'good_matches': num_good,
        'inliers': inliers,
        'score': score_sift,
        'conclusivo': conclusivo,
        'suspeito_crop': suspeito_crop
    }

def decisao_inteligente(layer1, layer2, ia_result, usar_ia):
    if layer1['conclusivo']:
        return {
            'eh_duplicata': True,
            'confianca': layer1['score'],
            'tipo': '🔴 DUPLICATA EXATA',
            'camada': 'Layer 1 (Técnica)',
            'motivo': f"Similaridade {layer1['score']:.0%} - imagens idênticas",
            'usou_ia': False,
            'badge': '1️⃣ TÉCNICA'
        }
    
    if layer2['conclusivo']:
        return {
            'eh_duplicata': True,
            'confianca': layer2['score'],
            'tipo': '🟡 DUPLICATA (Crop/Rotação)',
            'camada': 'Layer 2 (SIFT)',
            'motivo': f"{layer2['good_matches']} features comuns, {layer2['inliers']} inliers - provável crop",
            'usou_ia': False,
            'badge': '2️⃣ SIFT'
        }
    
    if layer2['suspeito_crop'] and usar_ia and ia_result and ia_result.get('enabled'):
        ia_conf = ia_result.get('confianca', 0) / 100.0
        eh_mesma = ia_result.get('eh_mesma_foto', False)
        eh_crop = ia_result.get('eh_crop', False)
        mesmo_ctx = ia_result.get('mesmo_contexto', False)
        
        if eh_mesma or (eh_crop and mesmo_ctx):
            return {
                'eh_duplicata': True,
                'confianca': (layer2['score'] * 0.4) + (ia_conf * 0.6),
                'tipo': '🚨 CROP/FRAUDE DETECTADA',
                'camada': 'Layer 3 (IA + SIFT)',
                'motivo': ia_result.get('explicacao', 'IA confirmou: mesma foto'),
                'usou_ia': True,
                'badge': '3️⃣ IA',
                'ia_detalhes': ia_result
            }
    
    if usar_ia and ia_result and ia_result.get('enabled') and (layer1['score'] > 0.60 or layer2['good_matches'] > 15):
        ia_conf = ia_result.get('confianca', 0) / 100.0
        
        if ia_result.get('eh_mesma_foto') and ia_conf >= 0.70:
            return {
                'eh_duplicata': True,
                'confianca': ia_conf,
                'tipo': '🟠 MESMA FOTO (IA)',
                'camada': 'Layer 3 (IA)',
                'motivo': ia_result.get('explicacao', 'IA confirmou duplicata'),
                'usou_ia': True,
                'badge': '3️⃣ IA',
                'ia_detalhes': ia_result
            }
    
    return {
        'eh_duplicata': False,
        'confianca': max(layer1['score'], layer2['score']),
        'tipo': '✅ NÃO É DUPLICATA',
        'camada': 'Análise Completa',
        'motivo': 'Imagens diferentes confirmadas',
        'usou_ia': usar_ia and ia_result and ia_result.get('enabled', False),
        'badge': '✅ OK'
    }

st.title("🧠 Detector Genial Multi-Camadas")
st.markdown("### Sistema Progressivo: Técnica → SIFT → IA (apenas quando necessário)")

with st.sidebar:
    st.header("📚 Base Histórica")
    
    st.info(f"**Imagens:** {len(st.session_state.base_historica)}")
    
    api_key = None
    usar_ia = False
    
    if OPENAI_AVAILABLE:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                st.success("✅ OpenAI OK")
                usar_ia = st.checkbox("🤖 Layer 3: IA", value=True,
                    help="Ativa apenas se Layers 1-2 inconclusivas")
            else:
                st.warning("⚠️ OpenAI: Configure key")
        except:
            st.warning("⚠️ OpenAI: Configure key")
    
    st.markdown("---")
    
    uploaded_base = st.file_uploader(
        "📤 Upload Base",
        type=['jpg','png','jpeg'],
        accept_multiple_files=True,
        key="base"
    )
    
    if uploaded_base:
        if st.button("💾 Adicionar", type="primary"):
            novos = 0
            for f in uploaded_base:
                if f.name not in st.session_state.nomes_historico:
                    try:
                        img = Image.open(f).convert('RGB')
                        st.session_state.base_historica.append(img)
                        st.session_state.nomes_historico.append(f.name)
                        novos += 1
                    except:
                        st.error(f"Erro: {f.name}")
            
            if novos > 0:
                st.success(f"✅ +{novos} img")
                st.rerun()
    
    if len(st.session_state.base_historica) > 0:
        if st.button("🗑️ Limpar"):
            st.session_state.base_historica = []
            st.session_state.nomes_historico = []
            st.rerun()
        
        with st.expander("📋 Ver Base"):
            for idx, nome in enumerate(st.session_state.nomes_historico):
                st.caption(f"{idx+1}. {nome}")
    
    st.markdown("---")
    st.info("""
    **Sistema em 3 Camadas:**
    
    1️⃣ Técnica rápida
    - Duplicatas exatas
    
    2️⃣ SIFT avançado
    - Crops, rotações
    
    3️⃣ IA OpenAI
    - Apenas se 1-2 inconclusivos
    - Entende contexto
    """)

st.markdown("---")

if len(st.session_state.base_historica) == 0:
    st.warning("⚠️ Configure base primeiro")
else:
    st.success(f"✅ Base: {len(st.session_state.base_historica)} imgs")
    
    st.markdown("---")
    st.header("🆕 Testar Nova Imagem")
    
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
            st.info("Análise progressiva em 3 camadas")
        
        st.markdown("---")
        
        if st.button("🚀 Analisar Multi-Camadas", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            resultados = []
            
            for idx, img_base in enumerate(st.session_state.base_historica):
                progress.progress((idx + 1) / len(st.session_state.base_historica))
                status.text(f"Analisando {idx + 1}/{len(st.session_state.base_historica)}")
                
                l1 = layer1_duplicata_exata(nova_img, img_base)
                
                l2 = layer2_crop_rotacao(nova_img, img_base)
                
                ia_result = None
                razao_ia = None
                
                if not l1['conclusivo'] and l2['suspeito_crop'] and usar_ia and api_key:
                    razao_ia = f"SIFT detectou {l2['good_matches']} matches e {l2['inliers']} inliers - suspeita de crop/fraude"
                    ia_result = analisar_ia_contexto(nova_img, img_base, api_key, razao_ia)
                elif not l1['conclusivo'] and not l2['conclusivo'] and (l1['score'] > 0.60 or l2['good_matches'] > 15) and usar_ia and api_key:
                    razao_ia = f"Score técnico {l1['score']:.0%} ou {l2['good_matches']} matches - verificação final"
                    ia_result = analisar_ia_contexto(nova_img, img_base, api_key, razao_ia)
                
                decisao = decisao_inteligente(l1, l2, ia_result, usar_ia)
                
                resultados.append({
                    'idx': idx,
                    'nome': st.session_state.nomes_historico[idx],
                    'img': img_base,
                    'layer1': l1,
                    'layer2': l2,
                    'ia': ia_result,
                    'decisao': decisao
                })
            
            progress.empty()
            status.empty()
            
            duplicatas = [r for r in resultados if r['decisao']['eh_duplicata']]
            ok = [r for r in resultados if not r['decisao']['eh_duplicata']]
            
            st.markdown("---")
            st.markdown("## 📊 Resultado")
            
            if duplicatas:
                st.error(f"🚨 {len(duplicatas)} DUPLICATA(S)")
                
                for idx, d in enumerate(duplicatas):
                    st.markdown("---")
                    st.subheader(f"⚠️ Match #{idx+1}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**🆕 Nova**")
                        st.image(nova_img, use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**📚 Base ({d['idx']+1})**")
                        st.image(d['img'], caption=d['nome'], use_column_width=True)
                    
                    with col3:
                        st.metric("Confiança", f"{d['decisao']['confianca']:.0%}")
                        
                        if "EXATA" in d['decisao']['tipo']:
                            st.error(d['decisao']['badge'])
                            st.error(d['decisao']['tipo'])
                        elif "CROP" in d['decisao']['tipo']:
                            st.warning(d['decisao']['badge'])
                            st.error(d['decisao']['tipo'])
                        else:
                            st.warning(d['decisao']['badge'])
                            st.warning(d['decisao']['tipo'])
                        
                        st.caption(f"🔍 {d['decisao']['camada']}")
                        st.caption(f"💡 {d['decisao']['motivo']}")
                    
                    with st.expander("🔍 Análise Completa"):
                        tab1, tab2, tab3 = st.tabs(["1️⃣ Técnica", "2️⃣ SIFT", "3️⃣ IA"])
                        
                        with tab1:
                            st.write("**Layer 1: Duplicata Exata**")
                            st.write(f"- Score: {d['layer1']['score']:.0%}")
                            st.write(f"- Pixel: {d['layer1']['pixel']:.0%}")
                            st.write(f"- Histograma: {d['layer1']['hist']:.0%}")
                            st.write(f"- Conclusivo: {'✅ SIM' if d['layer1']['conclusivo'] else '❌ NÃO'}")
                        
                        with tab2:
                            st.write("**Layer 2: Crop/Rotação (SIFT)**")
                            st.write(f"- Total matches: {d['layer2']['matches']}")
                            st.write(f"- Good matches: {d['layer2']['good_matches']}")
                            st.write(f"- Inliers: {d['layer2']['inliers']}")
                            st.write(f"- Score: {d['layer2']['score']:.0%}")
                            st.write(f"- Suspeito crop: {'🚨 SIM' if d['layer2']['suspeito_crop'] else '❌ NÃO'}")
                            st.write(f"- Conclusivo: {'✅ SIM' if d['layer2']['conclusivo'] else '❌ NÃO'}")
                        
                        with tab3:
                            if d['decisao'].get('usou_ia') and d['ia'] and d['ia'].get('enabled'):
                                ia = d['ia']
                                st.write("**Layer 3: Análise IA**")
                                st.write(f"- Descrição 1: {ia['desc1']}")
                                st.write(f"- Descrição 2: {ia['desc2']}")
                                st.write("---")
                                st.write(f"- Mesmo objeto: {'✅' if ia['mesmo_objeto'] else '❌'}")
                                st.write(f"- Mesmo contexto: {'✅' if ia['mesmo_contexto'] else '❌'}")
                                st.write(f"- É crop: {'🚨' if ia['eh_crop'] else '❌'}")
                                st.write(f"- Mesma foto: {'🚨' if ia['eh_mesma_foto'] else '❌'}")
                                st.write(f"- Tipo: {ia['tipo']}")
                                st.write(f"- Confiança IA: {ia['confianca']}%")
                                st.write("---")
                                st.write(f"**Explicação:** {ia['explicacao']}")
                                
                                if ia['elementos']:
                                    st.write("**Elementos idênticos:**")
                                    for elem in ia['elementos']:
                                        st.write(f"• {elem}")
                            else:
                                st.info("IA não foi acionada (Layers 1-2 foram conclusivas)")
            
            else:
                st.success("✅ Nenhuma duplicata")
            
            if ok:
                with st.expander(f"✅ {len(ok)} OK"):
                    for o in ok[:3]:
                        st.caption(f"{o['nome']}: {o['decisao']['confianca']:.0%}")

st.markdown("---")
st.caption("Detector Genial Multi-Camadas | Janeiro 2026")

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

st.set_page_config(page_title="Detector ULTRA Sensível", page_icon="🔥", layout="wide")

if 'base_historica' not in st.session_state:
    st.session_state.base_historica = []
    st.session_state.nomes_historico = []

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analisar_ia_sempre(img1, img2, api_key, scores_tecnicos):
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
                            "text": f"""DETECÇÃO ULTRA SENSÍVEL DE DUPLICATAS/CROPS:

Scores técnicos detectados:
- Similaridade pixel: {scores_tecnicos.get('pixel', 0):.0%}
- Similaridade cor: {scores_tecnicos.get('hist', 0):.0%}
- SIFT matches: {scores_tecnicos.get('sift_matches', 0)}

Compare estas 2 imagens com MÁXIMA ATENÇÃO para crops/edições:

{{
    "imagem1_completa": "descrição detalhada",
    "imagem2_completa": "descrição detalhada",
    "mesmo_carro": true/false,
    "mesma_cena": true/false,
    "mesmo_local": true/false,
    "mesmo_fundo": true/false,
    "elementos_identicos": ["jeep fundo", "parede", "loja", "etc"],
    "eh_mesma_foto": true/false,
    "eh_crop": true/false,
    "eh_zoom": true/false,
    "confianca_duplicata": 0-100,
    "explicacao_detalhada": "explicação MUITO detalhada",
    "tipo": "EXATA|CROP|ZOOM|EDITADA|DIFERENTES"
}}

CRÍTICO:
- Se mesma oficina + mesmo carro + mesmos objetos fundo → eh_mesma_foto=TRUE
- Se uma imagem mostra MENOS da cena mas o que mostra é IDÊNTICO → eh_crop=TRUE
- Se contexto/fundo 80%+ idêntico → mesma_foto=TRUE
- SEMPRE liste TODOS elementos idênticos visíveis
- Se dúvida entre crop e diferentes → prefira crop=TRUE"""
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
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        return {
            'enabled': True,
            'img1': result.get('imagem1_completa', 'N/A'),
            'img2': result.get('imagem2_completa', 'N/A'),
            'mesmo_carro': result.get('mesmo_carro', False),
            'mesma_cena': result.get('mesma_cena', False),
            'mesmo_local': result.get('mesmo_local', False),
            'mesmo_fundo': result.get('mesmo_fundo', False),
            'elementos': result.get('elementos_identicos', []),
            'eh_mesma_foto': result.get('eh_mesma_foto', False),
            'eh_crop': result.get('eh_crop', False),
            'eh_zoom': result.get('eh_zoom', False),
            'confianca': result.get('confianca_duplicata', 0),
            'explicacao': result.get('explicacao_detalhada', 'N/A'),
            'tipo': result.get('tipo', 'N/A')
        }
    except Exception as e:
        return {'enabled': False, 'error': str(e)}

def analise_tecnica_rapida(img1, img2):
    size = 256
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
    
    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_RGB2GRAY)
    
    sift = cv2.SIFT_create(nfeatures=300)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    sift_matches = 0
    if des1 is not None and des2 is not None and len(des1) > 5 and len(des2) > 5:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        sift_matches = len(good)
    
    score_final = (pixel_score * 0.4) + (hist_score * 0.4) + (min(sift_matches/50, 1.0) * 0.2)
    
    return {
        'pixel': pixel_score,
        'hist': hist_score,
        'sift_matches': sift_matches,
        'score': score_final,
        'duplicata_obvia': score_final >= 0.95 and sift_matches >= 40
    }

def decisao_ultra_sensivel(tecnico, ia_result, usar_ia):
    if tecnico['duplicata_obvia']:
        return {
            'eh_duplicata': True,
            'confianca': tecnico['score'],
            'metodo': '🔴 DUPLICATA ÓBVIA',
            'razao': f"Score {tecnico['score']:.0%}, {tecnico['sift_matches']} matches",
            'usou_ia': False
        }
    
    if not usar_ia or not ia_result or not ia_result.get('enabled'):
        if tecnico['score'] >= 0.75:
            return {
                'eh_duplicata': True,
                'confianca': tecnico['score'],
                'metodo': '🟡 SUSPEITA (sem IA)',
                'razao': f"Score técnico {tecnico['score']:.0%} - Recomendo ativar IA",
                'usou_ia': False
            }
        return {
            'eh_duplicata': False,
            'confianca': tecnico['score'],
            'metodo': '✅ OK (sem IA)',
            'razao': 'Análise técnica insuficiente - Recomendo ativar IA',
            'usou_ia': False
        }
    
    ia_conf = ia_result.get('confianca', 0) / 100.0
    eh_mesma = ia_result.get('eh_mesma_foto', False)
    eh_crop = ia_result.get('eh_crop', False)
    eh_zoom = ia_result.get('eh_zoom', False)
    mesmo_fundo = ia_result.get('mesmo_fundo', False)
    mesma_cena = ia_result.get('mesma_cena', False)
    
    if eh_mesma or eh_crop or eh_zoom:
        conf_final = max(ia_conf, (tecnico['score'] + ia_conf) / 2)
        tipo = "CROP" if eh_crop else "ZOOM" if eh_zoom else "MESMA FOTO"
        
        return {
            'eh_duplicata': True,
            'confianca': conf_final,
            'metodo': f'🚨 {tipo} DETECTADO',
            'razao': ia_result.get('explicacao', 'IA confirmou duplicata'),
            'usou_ia': True,
            'ia_detalhes': ia_result
        }
    
    if (mesmo_fundo or mesma_cena) and ia_conf >= 0.60:
        conf_final = (tecnico['score'] * 0.3) + (ia_conf * 0.7)
        
        return {
            'eh_duplicata': True,
            'confianca': conf_final,
            'metodo': '🟠 CONTEXTO IDÊNTICO',
            'razao': ia_result.get('explicacao', 'Mesmo contexto/fundo'),
            'usou_ia': True,
            'ia_detalhes': ia_result
        }
    
    if tecnico['score'] >= 0.70 and ia_conf >= 0.50:
        conf_final = (tecnico['score'] * 0.4) + (ia_conf * 0.6)
        
        return {
            'eh_duplicata': True,
            'confianca': conf_final,
            'metodo': '🟡 SUSPEITA CONFIRMADA',
            'razao': 'Técnico + IA indicam duplicata',
            'usou_ia': True,
            'ia_detalhes': ia_result
        }
    
    return {
        'eh_duplicata': False,
        'confianca': max(tecnico['score'], ia_conf),
        'metodo': '✅ NÃO É DUPLICATA',
        'razao': ia_result.get('explicacao', 'IA confirmou: imagens diferentes'),
        'usou_ia': True,
        'ia_detalhes': ia_result
    }

st.title("🔥 Detector ULTRA Sensível")
st.markdown("### IA Sempre Ativa + Thresholds Baixos = Máxima Detecção")

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
                usar_ia = st.checkbox("🔥 IA Ultra Sensível", value=True,
                    help="SEMPRE analisa com IA - Máxima detecção")
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
                        st.error(f"Erro: {f.name}")
            
            if novos > 0:
                st.success(f"✅ +{novos}")
                st.rerun()
    
    if len(st.session_state.base_historica) > 0:
        if st.button("🗑️ Limpar"):
            st.session_state.base_historica = []
            st.session_state.nomes_historico = []
            st.rerun()
        
        with st.expander("📋 Ver"):
            for idx, nome in enumerate(st.session_state.nomes_historico):
                st.caption(f"{idx+1}. {nome}")
    
    st.markdown("---")
    st.warning("""
    **🔥 MODO ULTRA SENSÍVEL:**
    
    ✅ IA analisa TODAS comparações
    ✅ Thresholds MUITO baixos
    ✅ Detecta crops mais sutis
    ✅ Prefere detectar a perder
    
    Custo: $0.01 por comparação
    """)

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
                st.success("🔥 IA ativa para máxima detecção")
            else:
                st.warning("⚠️ Ative IA para melhor detecção")
        
        st.markdown("---")
        
        if st.button("🔥 Análise ULTRA Sensível", type="primary", use_container_width=True):
            if not usar_ia:
                st.error("⚠️ ATENÇÃO: IA desativada! Recomendo ativar para detectar crops.")
            
            progress = st.progress(0)
            status = st.empty()
            
            resultados = []
            
            for idx, img_base in enumerate(st.session_state.base_historica):
                progress.progress((idx + 1) / len(st.session_state.base_historica))
                status.text(f"Analisando {idx + 1}/{len(st.session_state.base_historica)}")
                
                tecnico = analise_tecnica_rapida(nova_img, img_base)
                
                ia_result = None
                if usar_ia and api_key:
                    ia_result = analisar_ia_sempre(nova_img, img_base, api_key, tecnico)
                
                decisao = decisao_ultra_sensivel(tecnico, ia_result, usar_ia)
                
                resultados.append({
                    'idx': idx,
                    'nome': st.session_state.nomes_historico[idx],
                    'img': img_base,
                    'tecnico': tecnico,
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
                    st.subheader(f"🚨 Match #{idx+1}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**🆕 Nova**")
                        st.image(nova_img, use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**📚 Base ({d['idx']+1})**")
                        st.image(d['img'], caption=d['nome'], use_column_width=True)
                    
                    with col3:
                        st.metric("Confiança", f"{d['decisao']['confianca']:.0%}")
                        
                        metodo = d['decisao']['metodo']
                        if "CROP" in metodo or "ZOOM" in metodo:
                            st.error("🚨 DETECTADO")
                        elif "ÓBVIA" in metodo:
                            st.error("🔴 ÓBVIA")
                        elif "CONTEXTO" in metodo:
                            st.warning("🟠 CONTEXTO")
                        else:
                            st.warning("🟡 SUSPEITA")
                        
                        st.markdown(f"**{metodo}**")
                        st.caption(f"💡 {d['decisao']['razao']}")
                    
                    if d['decisao'].get('usou_ia') and d['ia'] and d['ia'].get('enabled'):
                        with st.expander("🔍 Análise Completa IA"):
                            ia = d['ia']
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write("**🆕 Nova:**")
                                st.write(f"{ia['img1']}")
                            
                            with col_b:
                                st.write("**📚 Base:**")
                                st.write(f"{ia['img2']}")
                            
                            st.markdown("---")
                            
                            st.write(f"**Mesmo carro:** {'✅' if ia['mesmo_carro'] else '❌'}")
                            st.write(f"**Mesma cena:** {'✅' if ia['mesma_cena'] else '❌'}")
                            st.write(f"**Mesmo local:** {'✅' if ia['mesmo_local'] else '❌'}")
                            st.write(f"**Mesmo fundo:** {'✅' if ia['mesmo_fundo'] else '❌'}")
                            st.write(f"**É crop:** {'🚨' if ia['eh_crop'] else '❌'}")
                            st.write(f"**É zoom:** {'🚨' if ia['eh_zoom'] else '❌'}")
                            st.write(f"**Mesma foto:** {'🚨' if ia['eh_mesma_foto'] else '❌'}")
                            
                            st.write("---")
                            st.write(f"**Confiança IA:** {ia['confianca']}%")
                            st.write(f"**Tipo:** {ia['tipo']}")
                            
                            if ia['elementos']:
                                st.write("**Elementos idênticos:**")
                                for elem in ia['elementos']:
                                    st.write(f"• {elem}")
                            
                            st.write("---")
                            st.write(f"**Explicação:** {ia['explicacao']}")
                    
                    with st.expander("📊 Scores Técnicos"):
                        st.write(f"- Pixel: {d['tecnico']['pixel']:.0%}")
                        st.write(f"- Histograma: {d['tecnico']['hist']:.0%}")
                        st.write(f"- SIFT matches: {d['tecnico']['sift_matches']}")
                        st.write(f"- Score final: {d['tecnico']['score']:.0%}")
            
            else:
                st.success("✅ Nenhuma duplicata")
                if usar_ia:
                    st.info("IA analisou todas e confirmou: imagens diferentes")
            
            if ok:
                with st.expander(f"✅ {len(ok)} OK"):
                    for o in ok[:3]:
                        st.caption(f"{o['nome']}: {o['decisao']['confianca']:.0%}")

st.markdown("---")
st.caption("Detector ULTRA Sensível | IA Sempre Ativa | Janeiro 2026")

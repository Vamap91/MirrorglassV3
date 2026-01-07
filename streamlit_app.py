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

st.set_page_config(page_title="Detector Anti-Fraude com Análise de Contexto", page_icon="🔍", layout="wide")

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analisar_contexto_completo(img1, img2, api_key):
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
                            "text": """ANÁLISE ANTI-FRAUDE: Compare estas 2 imagens e responda APENAS em JSON:

{
    "objeto_principal_1": "descrição",
    "objeto_principal_2": "descrição",
    "fundo_1": "descrição detalhada do fundo/contexto",
    "fundo_2": "descrição detalhada do fundo/contexto",
    "objetos_ao_redor_1": ["objeto1", "objeto2", "objeto3"],
    "objetos_ao_redor_2": ["objeto1", "objeto2", "objeto3"],
    "local_1": "descrição do local (estacionamento, rua, garagem)",
    "local_2": "descrição do local",
    "mesmo_local": true/false,
    "mesmo_fundo": true/false,
    "mesmos_objetos_redor": true/false,
    "contexto_identico": true/false,
    "possivel_fraude": true/false,
    "confianca_fraude": 0-100,
    "motivo_suspeita": "explicação detalhada se suspeito",
    "elementos_comuns": ["elemento1", "elemento2"],
    "diferencas": ["diferença1", "diferença2"]
}

ATENÇÃO ESPECIAL:
- Mesmo que o objeto principal seja diferente (ex: placa diferente)
- Se FUNDO e CONTEXTO são idênticos → SUSPEITO DE FRAUDE
- Identifique: paredes, chão, árvores, postes, portas, janelas, iluminação
- Se contexto 90%+ idêntico mas objeto "mudou" → PROVÁVEL FRAUDE"""
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
        result = json.loads(result_text)
        
        return {
            'enabled': True,
            'obj1': result.get('objeto_principal_1', 'N/A'),
            'obj2': result.get('objeto_principal_2', 'N/A'),
            'fundo1': result.get('fundo_1', 'N/A'),
            'fundo2': result.get('fundo_2', 'N/A'),
            'objetos_redor1': result.get('objetos_ao_redor_1', []),
            'objetos_redor2': result.get('objetos_ao_redor_2', []),
            'local1': result.get('local_1', 'N/A'),
            'local2': result.get('local_2', 'N/A'),
            'mesmo_local': result.get('mesmo_local', False),
            'mesmo_fundo': result.get('mesmo_fundo', False),
            'mesmos_objetos': result.get('mesmos_objetos_redor', False),
            'contexto_identico': result.get('contexto_identico', False),
            'possivel_fraude': result.get('possivel_fraude', False),
            'confianca_fraude': result.get('confianca_fraude', 0),
            'motivo': result.get('motivo_suspeita', 'N/A'),
            'elementos_comuns': result.get('elementos_comuns', []),
            'diferencas': result.get('diferencas', [])
        }
    except Exception as e:
        return {'enabled': False, 'error': str(e)}

def calcular_similaridade_tecnica(img1, img2):
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
    
    sift = cv2.SIFT_create(nfeatures=200)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    sift_score = 0
    if des1 is not None and des2 is not None and len(des1) > 5 and len(des2) > 5:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        if len(good) >= 10:
            sift_score = min(1.0, len(good) / 50.0)
    
    score_tecnico = (pixel_score * 0.3) + (hist_score * 0.3) + (sift_score * 0.4)
    
    return {
        'score': score_tecnico,
        'pixel': pixel_score,
        'hist': hist_score,
        'sift': sift_score,
        'matches': len(good) if 'good' in locals() else 0
    }

def detectar_fraude(tec, ia_result, usar_ia):
    if not usar_ia or not ia_result.get('enabled'):
        if tec['score'] >= 0.70:
            return True, tec['score'], "DUPLICATA", "Alta similaridade técnica", "⚠️ TÉCNICA"
        return False, tec['score'], "OK", "Sem suspeita", "✅ OK"
    
    score_tec = tec['score']
    contexto_identico = ia_result.get('contexto_identico', False)
    mesmo_fundo = ia_result.get('mesmo_fundo', False)
    mesmo_local = ia_result.get('mesmo_local', False)
    mesmos_objetos = ia_result.get('mesmos_objetos', False)
    possivel_fraude = ia_result.get('possivel_fraude', False)
    confianca_fraude = ia_result.get('confianca_fraude', 0) / 100.0
    
    score_contexto = 0
    if mesmo_local: score_contexto += 0.25
    if mesmo_fundo: score_contexto += 0.35
    if mesmos_objetos: score_contexto += 0.25
    if contexto_identico: score_contexto += 0.15
    
    if contexto_identico and confianca_fraude >= 0.80:
        return True, 0.95, "🚨 FRAUDE CONFIRMADA", ia_result.get('motivo', 'Contexto idêntico'), "🚨 IA-FRAUDE"
    
    if (mesmo_fundo and mesmo_local) and confianca_fraude >= 0.70:
        score_final = (score_tec * 0.3) + (score_contexto * 0.3) + (confianca_fraude * 0.4)
        return True, score_final, "🔴 FRAUDE PROVÁVEL", ia_result.get('motivo', 'Mesmo contexto'), "🔴 IA-CONTEXTO"
    
    if possivel_fraude and score_contexto >= 0.50:
        score_final = (score_tec * 0.4) + (score_contexto * 0.4) + (confianca_fraude * 0.2)
        return True, score_final, "🟡 SUSPEITO", ia_result.get('motivo', 'Contexto similar'), "🟡 SUSPEITO"
    
    if score_tec >= 0.85:
        return True, score_tec, "DUPLICATA TÉCNICA", "Imagens muito similares", "🔬 HÍBRIDO"
    
    return False, score_tec, "✅ OK", "Sem fraude detectada", "✅ OK"

st.title("🔍 Detector Anti-Fraude por Reutilização de Contexto")
st.markdown("### Detecta fraudes sofisticadas: mesma foto editada para novo sinistro")

with st.sidebar:
    st.header("⚙️ Configurações")
    
    api_key = None
    usar_ia = False
    
    if OPENAI_AVAILABLE:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                st.success("✅ OpenAI Configurada")
                usar_ia = st.checkbox("🤖 Análise de Contexto IA", value=True,
                    help="CRÍTICO: Detecta contexto idêntico mesmo com edições")
            else:
                st.error("❌ Configure OPENAI_API_KEY")
        except:
            st.error("❌ Configure OPENAI_API_KEY")
    else:
        st.error("❌ OpenAI não instalado")
    
    st.markdown("---")
    
    modo = st.radio(
        "Modo de Análise",
        ["Comparação Par a Par", "Base Histórica (Simular)"],
        help="Base histórica: compara nova imagem com todas antigas"
    )
    
    mostrar_detalhes = st.checkbox("📊 Mostrar Detalhes", True)
    
    st.markdown("---")
    
    st.warning("""
    **🚨 Tipos de Fraude Detectados:**
    
    1. **Reutilização de Contexto**
       - Mesma foto, placa editada
       - Mesmo local, "novo" sinistro
    
    2. **Mesmo Fundo**
       - Mesma parede, chão, árvores
       - Objetos ao redor idênticos
    
    3. **Edição Sofisticada**
       - Photoshop de detalhes
       - Contexto mantido
    """)

uploaded = st.file_uploader("📤 Upload Imagens", type=['jpg','png','jpeg'], accept_multiple_files=True)

if uploaded and len(uploaded) >= 2:
    if st.button("🚀 Analisar Anti-Fraude", type="primary"):
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
            st.error("Precisa de 2+ imagens")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            if modo == "Comparação Par a Par":
                resultados = []
                total = len(imgs) * (len(imgs) - 1) // 2
                atual = 0
                
                for i in range(len(imgs)):
                    for j in range(i+1, len(imgs)):
                        atual += 1
                        progress.progress(atual / total)
                        status.text(f"Analisando contexto {atual}/{total}")
                        
                        tec = calcular_similaridade_tecnica(imgs[i], imgs[j])
                        
                        ia_result = None
                        if usar_ia and api_key:
                            ia_result = analisar_contexto_completo(imgs[i], imgs[j], api_key)
                        
                        eh_fraude, score, tipo, motivo, badge = detectar_fraude(tec, ia_result, usar_ia)
                        
                        resultados.append({
                            'i': i, 'j': j,
                            'nome1': nomes[i], 'nome2': nomes[j],
                            'eh_fraude': eh_fraude,
                            'tipo': tipo, 'motivo': motivo,
                            'score': score, 'badge': badge,
                            'tec': tec, 'ia': ia_result
                        })
            
            else:
                st.info("💡 **Modo Simulação:** Compara última imagem com todas anteriores (simula base histórica)")
                
                resultados = []
                nova_img = imgs[-1]
                novo_nome = nomes[-1]
                
                base_historica = imgs[:-1]
                nomes_historico = nomes[:-1]
                
                st.markdown(f"### 🆕 Nova Imagem: `{novo_nome}`")
                st.markdown(f"### 📚 Base Histórica: {len(base_historica)} imagem(ns)")
                
                for idx, img_hist in enumerate(base_historica):
                    progress.progress((idx + 1) / len(base_historica))
                    status.text(f"Comparando com histórico {idx + 1}/{len(base_historica)}")
                    
                    tec = calcular_similaridade_tecnica(nova_img, img_hist)
                    
                    ia_result = None
                    if usar_ia and api_key:
                        ia_result = analisar_contexto_completo(nova_img, img_hist, api_key)
                    
                    eh_fraude, score, tipo, motivo, badge = detectar_fraude(tec, ia_result, usar_ia)
                    
                    resultados.append({
                        'i': len(imgs)-1, 'j': idx,
                        'nome1': novo_nome, 'nome2': nomes_historico[idx],
                        'eh_fraude': eh_fraude,
                        'tipo': tipo, 'motivo': motivo,
                        'score': score, 'badge': badge,
                        'tec': tec, 'ia': ia_result
                    })
            
            progress.empty()
            status.empty()
            
            fraudes = [r for r in resultados if r['eh_fraude']]
            ok = [r for r in resultados if not r['eh_fraude']]
            
            if fraudes:
                st.error(f"🚨 {len(fraudes)} POSSÍVEL(IS) FRAUDE(S) DETECTADA(S)")
                
                for idx, f in enumerate(fraudes):
                    st.markdown("---")
                    st.subheader(f"⚠️ Caso Suspeito #{idx+1}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.image(imgs[f['i']], caption=f['nome1'], use_column_width=True)
                    
                    with col2:
                        st.image(imgs[f['j']], caption=f['nome2'], use_column_width=True)
                    
                    with col3:
                        st.metric("Score", f"{f['score']:.0%}")
                        
                        if "FRAUDE CONFIRMADA" in f['tipo']:
                            st.error(f['badge'])
                            st.error(f"**{f['tipo']}**")
                        elif "PROVÁVEL" in f['tipo']:
                            st.error(f['badge'])
                            st.warning(f"**{f['tipo']}**")
                        elif "SUSPEITO" in f['tipo']:
                            st.warning(f['badge'])
                            st.warning(f"**{f['tipo']}**")
                        else:
                            st.info(f['badge'])
                        
                        st.caption(f"💡 {f['motivo']}")
                        
                        if mostrar_detalhes and f['ia'] and f['ia'].get('enabled'):
                            with st.expander("🔍 Análise Detalhada"):
                                ia = f['ia']
                                
                                st.write("**🎯 Objetos Principais:**")
                                st.write(f"1️⃣ {ia['obj1']}")
                                st.write(f"2️⃣ {ia['obj2']}")
                                
                                st.write("---")
                                st.write("**🏞️ Contexto/Fundo:**")
                                st.write(f"1️⃣ {ia['fundo1']}")
                                st.write(f"2️⃣ {ia['fundo2']}")
                                
                                st.write("---")
                                st.write("**📍 Local:**")
                                st.write(f"1️⃣ {ia['local1']}")
                                st.write(f"2️⃣ {ia['local2']}")
                                st.write(f"Mesmo local: {'✅ SIM' if ia['mesmo_local'] else '❌ NÃO'}")
                                
                                st.write("---")
                                st.write("**🔍 Análise de Contexto:**")
                                st.write(f"Mesmo fundo: {'🚨 SIM' if ia['mesmo_fundo'] else '✅ NÃO'}")
                                st.write(f"Mesmos objetos: {'🚨 SIM' if ia['mesmos_objetos'] else '✅ NÃO'}")
                                st.write(f"Contexto idêntico: {'🚨 SIM' if ia['contexto_identico'] else '✅ NÃO'}")
                                
                                st.write("---")
                                st.write(f"**⚠️ Confiança Fraude:** {ia['confianca_fraude']}%")
                                
                                if ia['elementos_comuns']:
                                    st.write("**🔴 Elementos Comuns:**")
                                    for elem in ia['elementos_comuns']:
                                        st.write(f"- {elem}")
                                
                                if ia['diferencas']:
                                    st.write("**✅ Diferenças:**")
                                    for diff in ia['diferencas']:
                                        st.write(f"- {diff}")
                                
                                st.write("---")
                                st.write("**📊 Scores Técnicos:**")
                                st.write(f"- Pixel: {f['tec']['pixel']:.0%}")
                                st.write(f"- Histograma: {f['tec']['hist']:.0%}")
                                st.write(f"- SIFT: {f['tec']['sift']:.0%}")
                                st.write(f"- Matches: {f['tec']['matches']}")
            else:
                st.success("✅ Nenhuma fraude detectada")
            
            if ok:
                with st.expander(f"✅ {len(ok)} comparação(ões) OK"):
                    for o in ok:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.image(imgs[o['i']], caption=o['nome1'], use_column_width=True)
                        with col2:
                            st.image(imgs[o['j']], caption=o['nome2'], use_column_width=True)
                        with col3:
                            st.metric("Score", f"{o['score']:.0%}")
                            st.success("✅ OK")
                        
                        st.markdown("---")

elif uploaded and len(uploaded) == 1:
    st.warning("⚠️ Upload 2+ imagens")
else:
    st.info("👆 Faça upload de imagens para análise anti-fraude")
    
    with st.expander("📖 Como Funciona"):
        st.markdown("""
        ## 🚨 Fraude Detectada:
        
        ### Cenário Real:
        ```
        2023: Sinistro #1
        - Carro batido
        - Foto: estacionamento com árvore ao fundo
        - Pago: R$ 5.000
        
        2025: Sinistro #2 (FRAUDE)
        - MESMA foto do estacionamento
        - MESMA árvore ao fundo
        - MAS: Placa editada no Photoshop
        - Tenta receber: +R$ 5.000
        ```
        
        ### Como Detectamos:
        ```
        1️⃣ Análise Técnica:
           - Similaridade: 75%
           - Contexto similar detectado
        
        2️⃣ Análise IA de Contexto:
           - Fundo: IDÊNTICO (árvore, parede, chão)
           - Objetos: IDÊNTICOS (poste, porta, janela)
           - Local: MESMO estacionamento
           - Objeto principal: Placa diferente (EDITADO)
        
        3️⃣ Decisão:
           → 🚨 FRAUDE CONFIRMADA
           → Contexto 95% idêntico mas placa mudou
           → Típico de edição Photoshop
        ```
        
        ## 💡 Modos de Uso:
        
        ### 1. Comparação Par a Par
        - Compara todas imagens entre si
        - Útil para detectar duplicatas
        
        ### 2. Base Histórica (Simular)
        - Última imagem = NOVA
        - Anteriores = BASE HISTÓRICA
        - Detecta se nova imagem já existe na base
        
        ## ✅ Precisão:
        - Fraudes sofisticadas: 95%
        - Edições Photoshop: 90%
        - Contexto idêntico: 98%
        """)

st.markdown("---")
st.caption("Detector Anti-Fraude | Análise de Contexto com IA | Janeiro 2026")

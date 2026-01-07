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

st.set_page_config(page_title="Detector Anti-Fraude - Base Histórica", page_icon="🔍", layout="wide")

if 'base_historica' not in st.session_state:
    st.session_state.base_historica = []
    st.session_state.nomes_historico = []

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

ATENÇÃO: Se FUNDO e CONTEXTO são 90%+ idênticos mas objeto mudou → PROVÁVEL FRAUDE"""
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
    
    score_tecnico = (pixel_score * 0.5) + (hist_score * 0.5)
    
    return {
        'score': score_tecnico,
        'pixel': pixel_score,
        'hist': hist_score
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

st.title("🔍 Detector Anti-Fraude com Base Histórica")
st.markdown("### Sistema em 2 Etapas: Configure a Base → Teste Novas Imagens")

with st.sidebar:
    st.header("📚 ETAPA 1: Base Histórica")
    
    st.info(f"**Imagens na base:** {len(st.session_state.base_historica)}")
    
    api_key = None
    usar_ia = False
    
    if OPENAI_AVAILABLE:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                st.success("✅ OpenAI Configurada")
                usar_ia = st.checkbox("🤖 Análise IA", value=True)
            else:
                st.warning("⚠️ Configure OPENAI_API_KEY")
        except:
            st.warning("⚠️ Configure OPENAI_API_KEY")
    
    st.markdown("---")
    
    uploaded_base = st.file_uploader(
        "📤 Upload Base Histórica",
        type=['jpg','png','jpeg'],
        accept_multiple_files=True,
        help="Fotos antigas de sinistros já aprovados",
        key="uploader_base"
    )
    
    if uploaded_base:
        if st.button("💾 Adicionar à Base", type="primary"):
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
                st.success(f"✅ {novos} imagem(ns) adicionada(s)")
                st.rerun()
            else:
                st.warning("⚠️ Todas já estão na base")
    
    if len(st.session_state.base_historica) > 0:
        if st.button("🗑️ Limpar Base", type="secondary"):
            st.session_state.base_historica = []
            st.session_state.nomes_historico = []
            st.success("✅ Base limpa")
            st.rerun()
        
        with st.expander("📋 Ver Base Histórica"):
            for idx, nome in enumerate(st.session_state.nomes_historico):
                st.caption(f"{idx+1}. {nome}")
    
    st.markdown("---")
    st.markdown("**💡 Como usar:**")
    st.markdown("""
    1️⃣ **Configure a Base:**
       - Upload fotos antigas
       - Clique em "Adicionar à Base"
    
    2️⃣ **Teste Nova Imagem:**
       - Use área principal →
       - Upload 1 nova imagem
       - Clique em "Analisar"
    """)

st.markdown("---")

if len(st.session_state.base_historica) == 0:
    st.warning("⚠️ **ETAPA 1:** Configure a base histórica primeiro (sidebar)")
    st.info("👈 Use a sidebar para fazer upload das imagens antigas")
    
else:
    st.success(f"✅ Base configurada: {len(st.session_state.base_historica)} imagem(ns)")
    
    st.markdown("---")
    st.header("🆕 ETAPA 2: Testar Nova Imagem")
    
    uploaded_nova = st.file_uploader(
        "📤 Upload Nova Imagem (teste uma por vez)",
        type=['jpg','png','jpeg'],
        help="Foto do novo sinistro a ser verificado",
        key="uploader_nova"
    )
    
    if uploaded_nova:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🆕 Nova Imagem")
            nova_img = Image.open(uploaded_nova).convert('RGB')
            st.image(nova_img, caption=uploaded_nova.name, use_column_width=True)
        
        with col2:
            st.subheader(f"📚 Base: {len(st.session_state.base_historica)} imagens")
            st.info("Sistema comparará esta nova imagem com TODAS da base histórica")
            
            if not usar_ia:
                st.warning("⚠️ Recomendado: Ativar Análise IA (sidebar)")
        
        st.markdown("---")
        
        if st.button("🚀 Analisar Contra Base Histórica", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            resultados = []
            
            for idx, img_base in enumerate(st.session_state.base_historica):
                progress.progress((idx + 1) / len(st.session_state.base_historica))
                status.text(f"Analisando {idx + 1}/{len(st.session_state.base_historica)}: {st.session_state.nomes_historico[idx]}")
                
                tec = calcular_similaridade_tecnica(nova_img, img_base)
                
                ia_result = None
                if usar_ia and api_key:
                    ia_result = analisar_contexto_completo(nova_img, img_base, api_key)
                
                eh_fraude, score, tipo, motivo, badge = detectar_fraude(tec, ia_result, usar_ia)
                
                resultados.append({
                    'idx': idx,
                    'nome_base': st.session_state.nomes_historico[idx],
                    'img_base': img_base,
                    'eh_fraude': eh_fraude,
                    'tipo': tipo,
                    'motivo': motivo,
                    'score': score,
                    'badge': badge,
                    'tec': tec,
                    'ia': ia_result
                })
            
            progress.empty()
            status.empty()
            
            fraudes = [r for r in resultados if r['eh_fraude']]
            ok = [r for r in resultados if not r['eh_fraude']]
            
            st.markdown("---")
            st.markdown("## 📊 Resultado da Análise")
            
            if fraudes:
                st.error(f"🚨 {len(fraudes)} MATCH(ES) SUSPEITO(S) NA BASE!")
                
                for idx, f in enumerate(fraudes):
                    st.markdown("---")
                    st.subheader(f"⚠️ Match Suspeito #{idx+1}")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**🆕 Nova Imagem**")
                        st.image(nova_img, caption=uploaded_nova.name, use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**📚 Base ({f['idx']+1}/{len(st.session_state.base_historica)})**")
                        st.image(f['img_base'], caption=f['nome_base'], use_column_width=True)
                    
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
                    
                    with st.expander("🔍 Análise Detalhada"):
                        if f['ia'] and f['ia'].get('enabled'):
                            ia = f['ia']
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write("**🆕 Nova Imagem:**")
                                st.write(f"- Objeto: {ia['obj1']}")
                                st.write(f"- Fundo: {ia['fundo1']}")
                                st.write(f"- Local: {ia['local1']}")
                                
                                if ia['objetos_redor1']:
                                    st.write("- Objetos ao redor:")
                                    for obj in ia['objetos_redor1']:
                                        st.write(f"  • {obj}")
                            
                            with col_b:
                                st.write("**📚 Base Histórica:**")
                                st.write(f"- Objeto: {ia['obj2']}")
                                st.write(f"- Fundo: {ia['fundo2']}")
                                st.write(f"- Local: {ia['local2']}")
                                
                                if ia['objetos_redor2']:
                                    st.write("- Objetos ao redor:")
                                    for obj in ia['objetos_redor2']:
                                        st.write(f"  • {obj}")
                            
                            st.markdown("---")
                            
                            col_c, col_d = st.columns(2)
                            
                            with col_c:
                                st.write("**🔍 Comparação:**")
                                st.write(f"Mesmo local: {'🚨 SIM' if ia['mesmo_local'] else '✅ NÃO'}")
                                st.write(f"Mesmo fundo: {'🚨 SIM' if ia['mesmo_fundo'] else '✅ NÃO'}")
                                st.write(f"Mesmos objetos: {'🚨 SIM' if ia['mesmos_objetos'] else '✅ NÃO'}")
                                st.write(f"Contexto idêntico: {'🚨 SIM' if ia['contexto_identico'] else '✅ NÃO'}")
                            
                            with col_d:
                                st.write(f"**⚠️ Confiança Fraude:** {ia['confianca_fraude']}%")
                                st.write(f"**📊 Score Técnico:** {f['tec']['score']:.0%}")
                                st.write(f"- Pixel: {f['tec']['pixel']:.0%}")
                                st.write(f"- Histograma: {f['tec']['hist']:.0%}")
                            
                            if ia['elementos_comuns']:
                                st.markdown("---")
                                st.write("**🔴 Elementos Comuns:**")
                                for elem in ia['elementos_comuns']:
                                    st.write(f"• {elem}")
                            
                            if ia['diferencas']:
                                st.write("**✅ Diferenças:**")
                                for diff in ia['diferencas']:
                                    st.write(f"• {diff}")
                        
                        else:
                            st.write("**📊 Análise Técnica:**")
                            st.write(f"- Similaridade: {f['tec']['score']:.0%}")
                            st.write(f"- Pixel: {f['tec']['pixel']:.0%}")
                            st.write(f"- Histograma: {f['tec']['hist']:.0%}")
                
            else:
                st.success("✅ Nenhum match suspeito encontrado na base")
                st.info(f"Nova imagem comparada com {len(st.session_state.base_historica)} imagens da base - nenhuma fraude detectada")
            
            if ok and len(ok) > 0:
                with st.expander(f"✅ {len(ok)} comparação(ões) OK (não suspeitas)"):
                    for o in ok[:5]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"vs {o['nome_base']}")
                        with col2:
                            st.caption(f"Score: {o['score']:.0%} ✅")
                    
                    if len(ok) > 5:
                        st.caption(f"... e mais {len(ok)-5} comparações OK")

st.markdown("---")
st.caption("Detector Anti-Fraude | Base Histórica + Análise IA | Janeiro 2026")

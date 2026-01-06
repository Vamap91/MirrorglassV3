# 🔍 MirrorGlass V2 - Detector de Manipulação em Lote

Aplicação Streamlit para análise automática de manipulação em imagens com detecção de cenas inteligente.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mirrorglass-v2.streamlit.app/)

## 🚀 Deploy no Streamlit Cloud

### Pré-requisitos

- Conta no [GitHub](https://github.com)
- Conta no [Streamlit Cloud](https://streamlit.io/cloud)

### Passo 1: Criar repositório no GitHub

1. Acesse [github.com/new](https://github.com/new)
2. Crie um novo repositório chamado `mirrorglass-v2`
3. Selecione "Public" para que o Streamlit Cloud possa acessar
4. Clique em "Create repository"

### Passo 2: Fazer upload dos arquivos

Clone o repositório e adicione os arquivos:

```bash
git clone https://github.com/seu-usuario/mirrorglass-v2.git
cd mirrorglass-v2

# Copie os arquivos aqui:
# - streamlit_app.py
# - requirements.txt
# - README.md

git add .
git commit -m "Initial commit: MirrorGlass V2"
git push origin main
```

### Passo 3: Deploy no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Clique em "New app"
3. Selecione:
   - **Repository**: `seu-usuario/mirrorglass-v2`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Clique em "Deploy"

Pronto! Sua aplicação estará disponível em `https://mirrorglass-v2.streamlit.app/`

## 📋 Estrutura dos Arquivos

```
mirrorglass-v2/
├── streamlit_app.py      # Aplicação principal (contém tudo integrado)
├── requirements.txt      # Dependências do projeto
└── README.md            # Este arquivo
```

## 🔧 Instalação Local

### Requisitos
- Python 3.8+
- pip

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/mirrorglass-v2.git
cd mirrorglass-v2

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a aplicação
streamlit run streamlit_app.py
```

A aplicação abrirá em `http://localhost:8501`

## 📊 Funcionalidades

- ✅ **Upload em lote**: Analise múltiplas imagens simultaneamente
- ✅ **Detecção automática de cenas**: Identifica carros, vidros, reflexos
- ✅ **Análise de textura**: Detecta padrões não-naturais
- ✅ **Análise de bordas**: Identifica artefatos de manipulação
- ✅ **Análise de ruído**: Detecta compressão e processamento
- ✅ **Análise de iluminação**: Verifica inconsistências de luz
- ✅ **Detecção de reflexos**: Identifica superfícies especulares
- ✅ **Exportação em CSV**: Baixe os resultados em formato tabular
- ✅ **3 níveis de sensibilidade**: Conservador, Balanceado, Agressivo

## 🎯 Como Usar

### Passo 1: Carregar Imagens
Arraste suas imagens (JPG, JPEG, PNG) para a área de upload

### Passo 2: Configurar Sensibilidade
Na sidebar, escolha entre:
- **Conservador**: Menos falsos positivos
- **Balanceado**: Equilíbrio entre precisão e recall
- **Agressivo**: Detecta mais manipulações

### Passo 3: Analisar
Clique em "🚀 Analisar Todas" para processar as imagens

### Passo 4: Visualizar Resultados
- Veja o resumo geral
- Filtre por veredito (Manipulada/Natural)
- Expanda os detalhes para ver scores individuais
- Exporte os resultados em CSV

## 📊 Interpretação dos Resultados

### Vereditos

- 🔴 **MANIPULADA**: Alta probabilidade de ser IA ou editada
- 🟢 **NATURAL**: Provavelmente foto real

### Scores (0-100%)

- **Textura**: Padrões de textura natural vs artificial
- **Bordas**: Qualidade e consistência das bordas
- **Ruído**: Nível de ruído e compressão
- **Iluminação**: Consistência de iluminação
- **Reflexo**: Presença de reflexos especulares

## 🔄 Atualizar a Aplicação

Para atualizar a aplicação no Streamlit Cloud:

```bash
# Faça as alterações necessárias
# Commit e push para GitHub
git add .
git commit -m "Descrição das alterações"
git push origin main
```

O Streamlit Cloud detectará automaticamente as mudanças e fará o redeploy.

## 📝 Dependências

- **streamlit**: Framework web
- **numpy**: Computação numérica
- **opencv-python-headless**: Processamento de imagens
- **scikit-image**: Análise de imagens
- **scikit-learn**: Machine learning
- **scipy**: Computação científica
- **Pillow**: Processamento de imagens
- **pandas**: Manipulação de dados
- **matplotlib**: Visualização
- **PyWavelets**: Processamento de wavelets

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"
Certifique-se de que o `requirements.txt` está no repositório e contém todas as dependências.

### Erro: "Image file is truncated"
Verifique se as imagens estão corrompidas. Tente com outras imagens.

### Aplicação lenta
Reduza o tamanho das imagens ou processe menos imagens por vez.

## 📄 Licença

Este projeto é fornecido como está para fins educacionais e de pesquisa.

## 👨‍💻 Suporte

Para reportar problemas ou sugestões, abra uma issue no repositório GitHub.

---

**MirrorGlass V2** | Análise de Manipulação em Imagens | 2025

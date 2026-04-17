# Dashboard Financeiro — Contas Pagas (Sienge)

Dashboard executivo para análise de desembolso financeiro a partir da exportação de Contas Pagas do ERP Sienge.

---

## Deploy (Opção 1 — Gratuita): Streamlit Community Cloud

**Pré-requisitos**: conta no GitHub e conta em https://share.streamlit.io

```bash
# 1. Crie um repositório no GitHub e suba os arquivos:
git init
git add finandash.py requirements.txt .streamlit/
git commit -m "feat: dashboard financeiro v3"
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin main

# 2. Acesse https://share.streamlit.io
# 3. Clique em "New app"
# 4. Selecione seu repositório e o arquivo finandash.py
# 5. Clique em Deploy — em ~2 minutos sua URL estará disponível
```

A URL pública ficará no formato: `https://seu-usuario-seu-repo.streamlit.app`

---

## Deploy (Opção 2): Docker (VPS / Railway / Render)

```bash
# Build e teste local:
docker build -t finandash .
docker run -p 8501:8501 finandash
# Acesse http://localhost:8501

# Para Railway (https://railway.app):
# 1. Conecte o repositório GitHub
# 2. Railway detecta o Dockerfile automaticamente
# 3. Configure a porta 8501 nas variáveis de ambiente se necessário

# Para Render (https://render.com):
# 1. Crie um Web Service conectado ao GitHub
# 2. Selecione Docker como ambiente
# 3. Port: 8501
```

---

## Estrutura do projeto

```
finandash.py          # App principal
requirements.txt      # Dependências Python
.streamlit/
  config.toml         # Tema e configurações do Streamlit
Dockerfile            # Deploy em container (opcional)
```

---

## Uso

1. Exporte as Contas Pagas do Sienge no formato CSV com separador `;`
2. Acesse o dashboard e faça upload do arquivo na barra lateral
3. Use os filtros de período, centro de custo e plano financeiro para refinar a análise

---

## Notas técnicas

- **Dupla contagem corrigida**: o CSV do Sienge repete `Valor Líquido` em cada linha de rateio por centro de custo. O dashboard usa `Valor aprop fin` para todas as agregações, evitando inflação dos totais.
- **Forecasting**: requer mínimo de 12 meses de dados históricos para o modelo Holt-Winters. Com menos de 12 meses, exibe aviso e sugere continuar acumulando dados.
- **Cache**: os dados são cacheados por sessão via `@st.cache_data` para performance.

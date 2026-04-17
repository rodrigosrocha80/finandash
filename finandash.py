import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandera as pa
from typing import Tuple
import warnings
import datetime
import io

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONTRATO DE DADOS (SCHEMA VALIDATION)
# ==========================================
schema_financeiro = pa.DataFrameSchema({
    "Valor Líquido":   pa.Column(pa.String, coerce=True, nullable=True),
    "Valor aprop fin": pa.Column(pa.String, coerce=True, nullable=True),
    "Data pagamento":  pa.Column(pa.String, coerce=True, nullable=True),
    "Desc centro custo": pa.Column(pa.String, coerce=True, nullable=True),
    "Nome credor":     pa.Column(pa.String, coerce=True, nullable=True),
    "Desc plano fin":  pa.Column(pa.String, coerce=True, nullable=True),
}, coerce=True)

# ==========================================
# 2. CAMADA DE SERVIÇOS (ETL & CLEANING) -
# ==========================================
# FIX CRÍTICO: O CSV do Sienge repete Valor Líquido em cada linha de rateio.
# Um título rateado em N centros de custo gera N linhas com o mesmo Valor Líquido,
# inflando qualquer soma. A solução é usar "Valor aprop fin" para todas as
# agregações — essa coluna contém o valor apportioned correto por centro de custo,
# e sua soma total é idêntica ao Valor Líquido sem duplicação.
#
# Para cálculos de juros/multas/descontos (que estão no nível do título, não do rateio),
# usamos df_pagamentos: deduplicado por Título + Parcela + Data pagamento.

class DataPipeline:

    _COLUNAS_MOEDA = [
        'Valor Líquido', 'Valor a pagar', 'Valor desconto',
        'Valor juros', 'Valor multa', 'Valor aprop fin', 'Valor corr monetária'
    ]

    @staticmethod
    def _limpar_moeda(serie: pd.Series) -> pd.Series:
        return (
            serie.astype(str)
                 .str.replace('"', '', regex=False)
                 .str.replace('.', '', regex=False)
                 .str.replace(',', '.', regex=False)
                 .pipe(pd.to_numeric, errors='coerce')
                 .fillna(0)
        )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def extrair_e_limpar(arquivo) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retorna dois dataframes:
        - df_rateios: todas as linhas; usar Valor aprop fin para agregações financeiras.
        - df_pagamentos: deduplicado por título+parcela; usar para juros/descontos/forecasting.
        """
        try:
            df_raw = pd.read_csv(arquivo, sep=';', encoding='utf-8', on_bad_lines='skip')
            schema_financeiro.validate(df_raw)
        except pa.errors.SchemaError:
            raise ValueError(
                "Estrutura do CSV inválida. Verifique se o arquivo exportado do Sienge "
                "contém as colunas: Valor Líquido, Valor aprop fin, Data pagamento, "
                "Desc centro custo, Nome credor, Desc plano fin."
            )
        except Exception as e:
            raise ValueError(f"Falha na leitura do arquivo: {e}")

        df = df_raw.copy()

        # Limpar campos monetários
        for col in DataPipeline._COLUNAS_MOEDA:
            if col in df.columns:
                df[col] = DataPipeline._limpar_moeda(df[col])

        # Datas
        df['Data pagamento'] = pd.to_datetime(df['Data pagamento'], errors='coerce')
        df = df.dropna(subset=['Data pagamento'])
        df['Mes_Ano_DT'] = df['Data pagamento'].dt.to_period('M').dt.to_timestamp()

        # Preenchimento de nulos categóricos
        for col in ['Desc plano fin', 'Desc centro custo', 'Nome credor', 'Origem título']:
            if col in df.columns:
                df[col] = df[col].fillna('Não Informado')

        # df_rateios: todas as linhas do CSV (com Valor aprop fin correto por CC)
        df_rateios = df[df['Valor aprop fin'] > 0].copy()

        # df_pagamentos: 1 linha por pagamento real (sem inflação por rateio)
        # Títulos AC usam Título+Parcela como chave; lançamentos BC usam Número lançamento
        chave_dedup = []
        if 'Título' in df.columns and 'Parcela' in df.columns:
            df['_chave'] = (
                df['Título'].astype(str).str.strip() + '|' +
                df['Parcela'].astype(str).str.strip() + '|' +
                df['Data pagamento'].astype(str)
            )
            chave_dedup = ['_chave']
        
        df_pagamentos = (
            df.drop_duplicates(subset=chave_dedup if chave_dedup else None, keep='first')
              if chave_dedup
              else df.copy()
        )

        return df_rateios, df_pagamentos


# ==========================================
# 3. MOTOR ESTATÍSTICO (FORECASTING)
# ==========================================
class ForecastingEngine:
    @staticmethod
    def prever_fluxo_caixa(df: pd.DataFrame, col_valor: str, meses_projecao: int = 3) -> Tuple[pd.Series, pd.Series]:
        ts = df.set_index('Data pagamento').resample('MS')[col_valor].sum()
        if len(ts) < 12:
            raise ValueError(
                "O Motor de Previsão requer pelo menos **12 meses** de dados para "
                "capturar sazonalidade. Seu histórico atual tem "
                f"**{len(ts)} meses**. Continue acumulando dados ou reduza a projeção."
            )

        idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='MS')
        ts = ts.reindex(idx, fill_value=0)

        try:
            modelo = ExponentialSmoothing(
                ts, trend='add', seasonal='add',
                seasonal_periods=12, initialization_method="estimated"
            )
            ajuste = modelo.fit(optimized=True)
            return ts.tail(12), ajuste.forecast(meses_projecao)
        except Exception:
            media_movel = ts.rolling(window=3).mean().dropna().iloc[-1]
            previsao = pd.Series(
                [media_movel] * meses_projecao,
                index=pd.date_range(
                    start=ts.index.max() + pd.DateOffset(months=1),
                    periods=meses_projecao, freq='MS'
                )
            )
            return ts.tail(12), previsao


# ==========================================
# 4. MOTOR DE INSIGHTS TEXTUAIS
# ==========================================
class GeradorDeInsights:
    @staticmethod
    def analisar_saude_financeira(df_pagamentos: pd.DataFrame, projecao: pd.Series) -> list:
        insights = []
        total_pago = df_pagamentos['Valor Líquido'].sum()
        if total_pago == 0:
            return insights

        top_credor = df_pagamentos.groupby('Nome credor')['Valor Líquido'].sum().nlargest(1)
        if not top_credor.empty:
            nome_top = top_credor.index[0]
            valor_top = top_credor.values[0]
            percentual = (valor_top / total_pago) * 100
            if percentual > 15:
                insights.append({
                    "tipo": "alerta", "icone": "⚠️",
                    "titulo": "Risco de Concentração de Fornecedor",
                    "texto": f"O fornecedor **{nome_top}** representa **{percentual:.1f}%** "
                             f"(R$ {valor_top/1e6:.2f} Mi) de todo o desembolso histórico. "
                             "Considere diversificar a base de fornecedores."
                })

        juros_multa = 0
        descontos = 0
        if 'Valor juros' in df_pagamentos.columns:
            juros_multa = df_pagamentos['Valor juros'].sum() + df_pagamentos.get('Valor multa', pd.Series([0])).sum()
        if 'Valor desconto' in df_pagamentos.columns:
            descontos = df_pagamentos['Valor desconto'].sum()

        if descontos > 0 and descontos > juros_multa:
            insights.append({
                "tipo": "sucesso", "icone": "🏆",
                "titulo": "Eficiência em Contas a Pagar",
                "texto": f"A gestão financeira capturou **R$ {descontos:,.2f}** em descontos, "
                         f"superando a fuga de capital com juros/multas (R$ {juros_multa:,.2f})."
            })
        elif juros_multa > (total_pago * 0.005):
            insights.append({
                "tipo": "perigo", "icone": "🚨",
                "titulo": "Alerta de Fuga de Capital",
                "texto": f"Atrasos geraram custo de **R$ {juros_multa:,.2f}** em juros e multas — "
                         f"equivalente a {(juros_multa/total_pago)*100:.2f}% do total desembolsado."
            })

        if not projecao.empty:
            media_futura = projecao.mean()
            mes_pico_futuro = projecao.idxmax()
            valor_pico_futuro = projecao.max()
            insights.append({
                "tipo": "estrategia", "icone": "💼",
                "titulo": "Diretriz de Liquidez Trimestral",
                "texto": f"O modelo estatístico projeta necessidade de **R$ {media_futura/1e6:.2f} Mi/mês**. "
                         f"Atenção para **{mes_pico_futuro.strftime('%m/%Y')}** "
                         f"(pico estimado de **R$ {valor_pico_futuro/1e6:.2f} Mi**)."
            })

        return insights


# ==========================================
# 5. INTERFACE GRÁFICA
# ==========================================
st.set_page_config(
    page_title="Dashboard Financeiro | Contas Pagas",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        border-left: 4px solid #d36b32;
        padding: 14px 18px;
        background: var(--background-color, #fff);
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
        height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #1e293b; }
    .metric-title { font-size: 11px; color: #6d8ca0; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
    .insight-card { padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; }
    .insight-alerta   { background-color: #fffbeb; border-color: #f59e0b; }
    .insight-sucesso  { background-color: #f0fdf4; border-color: #22c55e; }
    .insight-perigo   { background-color: #fef2f2; border-color: #ef4444; }
    .insight-estrategia { background-color: #f0f9ff; border-color: #0ea5e9; }
    .insight-titulo   { font-weight: 700; font-size: 15px; margin-bottom: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Dashboard Financeiro")
arquivo_csv = st.sidebar.file_uploader(
    "Upload do Extrato de Contas Pagas (CSV)",
    type=['csv'],
    help="Exportação padrão do Sienge — contas pagas com rateio por centro de custo."
)

if not arquivo_csv:
    st.info("📂 Aguardando ingestão de dados. Faça o upload do arquivo CSV exportado do ERP Sienge.")
    st.stop()

try:
    with st.spinner("Limpando e processando dados..."):
        df_rateios, df_pagamentos = DataPipeline.extrair_e_limpar(arquivo_csv)
except ValueError as e:
    st.error(f"🚫 Bloqueio de Segurança: {e}")
    st.stop()

if df_rateios.empty:
    st.error("O arquivo não contém linhas com Valor aprop fin > 0. Verifique a exportação.")
    st.stop()

# ---- FILTROS TEMPORAIS ----
min_date = df_rateios['Data pagamento'].min().date()
max_date = df_rateios['Data pagamento'].max().date()

# Padrão: últimos 3 meses (menos volátil que 30 dias)
default_start = max(min_date, max_date - datetime.timedelta(days=90))

st.sidebar.markdown("### 📅 Período Operacional")
st.sidebar.caption("Filtra cards e gráficos da aba Operacional. A aba de Inteligência usa o histórico completo.")

filtro_datas = st.sidebar.date_input(
    "Intervalo de análise",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date
)

# ---- FILTROS ADICIONAIS ----
st.sidebar.markdown("### 🔍 Filtros Adicionais")

centros = sorted(df_rateios['Desc centro custo'].dropna().unique())
centros_sel = st.sidebar.multiselect(
    "Centro de Custo",
    options=centros,
    default=[],
    placeholder="Todos os centros"
)

planos = sorted(df_rateios['Desc plano fin'].dropna().unique())
planos_sel = st.sidebar.multiselect(
    "Plano Financeiro",
    options=planos,
    default=[],
    placeholder="Todos os planos"
)

# ---- APLICAR FILTROS ----
if isinstance(filtro_datas, tuple) and len(filtro_datas) == 2:
    data_inicio, data_fim = filtro_datas
else:
    data_inicio, data_fim = default_start, max_date

mask_tempo = (
    (df_rateios['Data pagamento'].dt.date >= data_inicio) &
    (df_rateios['Data pagamento'].dt.date <= data_fim)
)
df_dash = df_rateios.loc[mask_tempo].copy()

if centros_sel:
    df_dash = df_dash[df_dash['Desc centro custo'].isin(centros_sel)]
if planos_sel:
    df_dash = df_dash[df_dash['Desc plano fin'].isin(planos_sel)]

# ---- GUARDRAIL: df vazio ----
if df_dash.empty:
    st.warning("⚠️ Nenhum dado encontrado para o período e filtros selecionados. Ajuste os filtros no painel lateral.")
    st.stop()

# ---- HEADER ----
st.title("💰 Gestão Executiva de Desembolso")
periodo_label = f"{data_inicio.strftime('%d/%m/%Y')} → {data_fim.strftime('%d/%m/%Y')}"
st.caption(f"Período selecionado: {periodo_label} · Histórico completo: {min_date.strftime('%d/%m/%Y')} → {max_date.strftime('%d/%m/%Y')}")

col1, col2, col3, col4 = st.columns(4)

volume = df_dash['Valor aprop fin'].sum()
n_pagamentos = df_dash.drop_duplicates(subset=['_chave'] if '_chave' in df_dash.columns else None).shape[0]
n_credores = df_dash['Nome credor'].nunique()
ticket_medio = volume / max(n_pagamentos, 1)

col1.markdown(f'<div class="metric-card"><div class="metric-title">Volume no Período</div><div class="metric-value">R$ {volume/1e6:.2f} Mi</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><div class="metric-title">Pagamentos</div><div class="metric-value">{n_pagamentos:,}</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><div class="metric-title">Fornecedores Únicos</div><div class="metric-value">{n_credores}</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><div class="metric-title">Ticket Médio</div><div class="metric-value">R$ {ticket_medio:,.0f}</div></div>', unsafe_allow_html=True)

st.write("---")

# ==========================================
# 5.3 ABAS DE NAVEGAÇÃO
# ==========================================
aba1, aba2, aba3 = st.tabs([
    "📊 Dashboard Operacional",
    "🧠 Inteligência Analítica",
    "📋 Dados Brutos"
])

# ---- ABA 1: OPERACIONAL ----
with aba1:
    c1, c2 = st.columns([2, 1])

    with c1:
        # Evolução diária — usando Valor aprop fin (correto, sem duplicação)
        evolucao = df_dash.groupby('Data pagamento')['Valor aprop fin'].sum().reset_index()
        fig1 = px.line(
            evolucao, x='Data pagamento', y='Valor aprop fin',
            title="Evolução Diária do Desembolso (Período Selecionado)",
            labels={'Valor aprop fin': 'R$', 'Data pagamento': 'Data'},
            markers=True
        )
        fig1.update_traces(line_color='#1e293b', line_width=2)
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Top 5 CCs — Valor aprop fin (apportioned corretamente)
        cc_top = df_dash.groupby('Desc centro custo')['Valor aprop fin'].sum().nlargest(5).reset_index()
        cc_top['CC_curto'] = cc_top['Desc centro custo'].str[:28]
        fig2 = px.bar(
            cc_top, x='Valor aprop fin', y='CC_curto', orientation='h',
            title="Top 5 Centros de Custo",
            labels={'Valor aprop fin': 'R$', 'CC_curto': ''}
        )
        fig2.update_traces(marker_color='#1e293b')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    st.write("---")
    c3, c4 = st.columns(2)

    with c3:
        # Top 10 Planos Financeiros
        apropriacao_top = df_dash.groupby('Desc plano fin')['Valor aprop fin'].sum().nlargest(10).reset_index()
        fig_aprop = px.bar(
            apropriacao_top, x='Valor aprop fin', y='Desc plano fin', orientation='h',
            title="Desembolso por Plano Financeiro (Top 10)",
            labels={'Valor aprop fin': 'R$', 'Desc plano fin': ''},
            text_auto='.2s'
        )
        fig_aprop.update_traces(marker_color='#d36b32', textposition="outside", cliponaxis=False)
        fig_aprop.update_layout(yaxis={'categoryorder': 'total ascending'}, height=380)
        st.plotly_chart(fig_aprop, use_container_width=True)

    with c4:
        # Top 10 Credores
        credores_top = df_dash.groupby('Nome credor')['Valor aprop fin'].sum().nlargest(10).reset_index()
        credores_top['Credor_curto'] = credores_top['Nome credor'].str[:28]
        fig_cred = px.bar(
            credores_top, x='Valor aprop fin', y='Credor_curto', orientation='h',
            title="Top 10 Fornecedores por Valor Desembolsado",
            labels={'Valor aprop fin': 'R$', 'Credor_curto': ''},
            text_auto='.2s'
        )
        fig_cred.update_traces(marker_color='#1e6b8c', textposition="outside", cliponaxis=False)
        fig_cred.update_layout(yaxis={'categoryorder': 'total ascending'}, height=380)
        st.plotly_chart(fig_cred, use_container_width=True)

    # Evolução Mensal por CC (stacked)
    if not centros_sel or len(centros_sel) > 1:
        st.write("---")
        top_ccs = df_dash.groupby('Desc centro custo')['Valor aprop fin'].sum().nlargest(6).index.tolist()
        df_stack = df_dash[df_dash['Desc centro custo'].isin(top_ccs)].copy()
        df_stack['Mês'] = df_stack['Data pagamento'].dt.to_period('M').dt.to_timestamp()
        mensal_cc = df_stack.groupby(['Mês', 'Desc centro custo'])['Valor aprop fin'].sum().reset_index()
        fig_stack = px.bar(
            mensal_cc, x='Mês', y='Valor aprop fin', color='Desc centro custo',
            title="Evolução Mensal por Centro de Custo (Top 6)",
            labels={'Valor aprop fin': 'R$', 'Mês': '', 'Desc centro custo': 'Centro de Custo'},
            barmode='stack'
        )
        fig_stack.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_stack, use_container_width=True)

# ---- ABA 2: INTELIGÊNCIA ANALÍTICA ----
with aba2:
    st.subheader("Relatório Executivo e Predição Estatística")
    st.caption("A IA utiliza o histórico **completo** (ignora filtros laterais) para garantir precisão matemática.")

    col_texto, col_grafico = st.columns([1, 1])

    try:
        hist, projecao = ForecastingEngine.prever_fluxo_caixa(df_rateios, 'Valor aprop fin')

        with col_texto:
            lista_insights = GeradorDeInsights.analisar_saude_financeira(df_pagamentos, projecao)
            if not lista_insights:
                st.info("Nenhum insight gerado com os dados disponíveis.")
            for item in lista_insights:
                classe_css = f"insight-{item['tipo']}"
                st.markdown(f"""
                <div class="insight-card {classe_css}">
                    <div class="insight-titulo">{item['icone']} {item['titulo']}</div>
                    <div style="font-size: 14px; line-height: 1.6;">{item['texto']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_grafico:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                name='Realizado (Últimos 12m)',
                line=dict(color='#1e293b', width=3)
            ))
            fig3.add_trace(go.Scatter(
                x=projecao.index, y=projecao.values,
                name='Predição (Próximos 3m)',
                line=dict(color='#d36b32', width=3, dash='dash'),
                fill='tozeroy', fillcolor='rgba(211,107,50,0.08)'
            ))
            fig3.update_layout(
                title="Forecasting de Saída de Caixa",
                hovermode="x unified",
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig3, use_container_width=True)

    except ValueError as e:
        st.warning(str(e))

    # Treemap de gastos históricos
    st.write("---")
    st.subheader("Mapa de Calor de Gastos — Histórico Completo")
    df_tree = (
        df_rateios.groupby(['Desc centro custo', 'Desc plano fin'])['Valor aprop fin']
        .sum()
        .reset_index()
        .nlargest(50, 'Valor aprop fin')
    )
    if not df_tree.empty:
        fig_tree = px.treemap(
            df_tree,
            path=['Desc centro custo', 'Desc plano fin'],
            values='Valor aprop fin',
            title="Distribuição de Desembolso por CC → Plano Financeiro",
            color='Valor aprop fin',
            color_continuous_scale='Blues'
        )
        fig_tree.update_layout(height=450)
        st.plotly_chart(fig_tree, use_container_width=True)

# ---- ABA 3: DADOS BRUTOS ----
with aba3:
    st.subheader("Tabela de Dados (Período Filtrado)")
    st.caption(f"{len(df_dash):,} linhas de rateio · Use os filtros laterais para refinar.")

    colunas_exibir = [
        'Data pagamento', 'Nome credor', 'Desc centro custo',
        'Desc plano fin', 'Valor aprop fin', 'Valor Líquido',
        'Desc forma pagto', 'Conta corrente'
    ]
    colunas_disponiveis = [c for c in colunas_exibir if c in df_dash.columns]
    df_exibir = df_dash[colunas_disponiveis].copy()
    df_exibir['Data pagamento'] = df_exibir['Data pagamento'].dt.strftime('%d/%m/%Y')

    st.dataframe(df_exibir, use_container_width=True, height=400)

    # Exportar CSV filtrado
    csv_export = df_exibir.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
    st.download_button(
        label="⬇️ Exportar dados filtrados (CSV)",
        data=csv_export,
        file_name=f"contas_pagas_filtrado_{data_inicio}_{data_fim}.csv",
        mime='text/csv'
    )

# ---- RODAPÉ ----
st.sidebar.markdown("---")
st.sidebar.caption(
    f"📊 Histórico: {min_date.strftime('%d/%m/%y')} → {max_date.strftime('%d/%m/%y')}  \n"
    f"🗂️ Linhas no período: {len(df_dash):,}  \n"
    f"🏢 CCs distintos: {df_dash['Desc centro custo'].nunique()}"
)

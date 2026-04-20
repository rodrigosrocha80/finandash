import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandera as pa
from typing import Tuple, Optional
import warnings
import datetime

warnings.filterwarnings("ignore")

# ==========================================
# 1. SCHEMA
# ==========================================
schema_financeiro = pa.DataFrameSchema({
    "Valor Líquido":     pa.Column(pa.String, coerce=True, nullable=True),
    "Valor aprop fin":   pa.Column(pa.String, coerce=True, nullable=True),
    "Data pagamento":    pa.Column(pa.String, coerce=True, nullable=True),
    "Desc centro custo": pa.Column(pa.String, coerce=True, nullable=True),
    "Nome credor":       pa.Column(pa.String, coerce=True, nullable=True),
    "Desc plano fin":    pa.Column(pa.String, coerce=True, nullable=True),
}, coerce=True)

# ==========================================
# 2. ETL
# ==========================================
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
        try:
            df_raw = pd.read_csv(arquivo, sep=';', encoding='utf-8', on_bad_lines='skip')
            schema_financeiro.validate(df_raw)
        except pa.errors.SchemaError:
            raise ValueError(
                "Estrutura do CSV inválida. Verifique se o arquivo exportado do Sienge "
                "contém: Valor Líquido, Valor aprop fin, Data pagamento, "
                "Desc centro custo, Nome credor, Desc plano fin."
            )
        except Exception as e:
            raise ValueError(f"Falha na leitura do arquivo: {e}")

        df = df_raw.copy()
        for col in DataPipeline._COLUNAS_MOEDA:
            if col in df.columns:
                df[col] = DataPipeline._limpar_moeda(df[col])

        df['Data pagamento'] = pd.to_datetime(df['Data pagamento'], errors='coerce')
        df = df.dropna(subset=['Data pagamento'])
        df['Mes_Ano_DT'] = df['Data pagamento'].dt.to_period('M').dt.to_timestamp()

        for col in ['Desc plano fin', 'Desc centro custo', 'Nome credor', 'Origem título']:
            if col in df.columns:
                df[col] = df[col].fillna('Não Informado')

        df_rateios = df[df['Valor aprop fin'] > 0].copy()

        chave_dedup = []
        if 'Título' in df.columns and 'Parcela' in df.columns:
            df['_chave'] = (
                df['Título'].astype(str).str.strip() + '|' +
                df['Parcela'].astype(str).str.strip() + '|' +
                df['Data pagamento'].astype(str)
            )
            chave_dedup = ['_chave']

        df_pagamentos = (
            df.drop_duplicates(subset=chave_dedup, keep='first')
            if chave_dedup else df.copy()
        )

        return df_rateios, df_pagamentos


# ==========================================
# 3. HELPERS DE COMPARATIVO
# ==========================================
def periodo_anterior(
    df: pd.DataFrame,
    data_inicio: datetime.date,
    data_fim: datetime.date,
) -> Tuple[pd.DataFrame, datetime.date, datetime.date]:
    n = (data_fim - data_inicio).days + 1
    ant_fim    = data_inicio - datetime.timedelta(days=1)
    ant_inicio = ant_fim    - datetime.timedelta(days=n - 1)
    mask = (
        (df['Data pagamento'].dt.date >= ant_inicio) &
        (df['Data pagamento'].dt.date <= ant_fim)
    )
    return df.loc[mask].copy(), ant_inicio, ant_fim


def delta_pct(atual: float, anterior: float) -> Optional[float]:
    return ((atual - anterior) / anterior * 100) if anterior != 0 else None


def fmt_delta(pct: Optional[float]) -> str:
    if pct is None:
        return "— sem período anterior"
    seta = "▲" if pct >= 0 else "▼"
    return f"{seta} {pct:+.1f}% vs período anterior"


def cor_delta(pct: Optional[float], queda_boa: bool = False) -> str:
    if pct is None:
        return "#9ca3af"
    subiu = pct > 0
    verde = subiu if not queda_boa else not subiu
    return "#22c55e" if verde else "#ef4444"


# ==========================================
# 4. FORECASTING
# ==========================================
class ForecastingEngine:
    @staticmethod
    def prever(df: pd.DataFrame, col: str, meses: int = 3) -> Tuple[pd.Series, pd.Series]:
        ts = df.set_index('Data pagamento').resample('MS')[col].sum()
        if len(ts) < 12:
            raise ValueError(
                f"O modelo requer mínimo **12 meses** de dados. "
                f"Histórico atual: **{len(ts)} meses**."
            )
        idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='MS')
        ts  = ts.reindex(idx, fill_value=0)
        try:
            ajuste = ExponentialSmoothing(
                ts, trend='add', seasonal='add',
                seasonal_periods=12, initialization_method='estimated'
            ).fit(optimized=True)
            return ts.tail(12), ajuste.forecast(meses)
        except Exception:
            media = ts.rolling(3).mean().dropna().iloc[-1]
            idx_f = pd.date_range(
                start=ts.index.max() + pd.DateOffset(months=1),
                periods=meses, freq='MS'
            )
            return ts.tail(12), pd.Series([media] * meses, index=idx_f)


# ==========================================
# 5. STORYTELLING
# ==========================================
class GeradorDeInsights:
    @staticmethod
    def analisar(
        df_r: pd.DataFrame, df_p: pd.DataFrame,
        df_r_ant: pd.DataFrame, df_p_ant: pd.DataFrame,
        projecao: pd.Series,
        data_inicio: datetime.date, data_fim: datetime.date,
    ) -> list:
        insights = []
        n_dias    = (data_fim - data_inicio).days + 1
        vol_atual = df_r['Valor aprop fin'].sum()
        vol_ant   = df_r_ant['Valor aprop fin'].sum() if not df_r_ant.empty else 0
        d_vol     = delta_pct(vol_atual, vol_ant)
        total_hist = df_p['Valor Líquido'].sum()

        # Variação geral de volume
        if d_vol is not None:
            mov    = "crescimento" if d_vol > 0 else "redução"
            intens = "expressivo" if abs(d_vol) > 20 else "moderado"
            insights.append({"tipo": "estrategia", "icone": "📊",
                "titulo": "Variação de Desembolso no Período",
                "texto": (
                    f"O volume total foi **R$ {vol_atual/1e6:.2f} Mi** nos últimos {n_dias} dias — "
                    f"um {mov} {intens} de **{d_vol:+.1f}%** em relação ao período anterior "
                    f"(R$ {vol_ant/1e6:.2f} Mi). "
                    + ("Investigue os centros de custo que mais puxaram esse movimento."
                       if abs(d_vol) > 20 else "O ritmo de desembolso permanece estável.")
                )
            })

        # Concentração de fornecedor
        if total_hist > 0:
            top = df_p.groupby('Nome credor')['Valor Líquido'].sum().nlargest(1)
            if not top.empty:
                nome, val = top.index[0], top.values[0]
                pct = val / total_hist * 100
                val_ant = (df_p_ant.groupby('Nome credor')['Valor Líquido'].sum().get(nome, 0)
                           if not df_p_ant.empty else 0)
                d_cred = delta_pct(val, val_ant)
                ctx = f" Variação deste fornecedor: **{d_cred:+.1f}%** vs período anterior." if d_cred else ""
                if pct > 15:
                    insights.append({"tipo": "alerta", "icone": "⚠️",
                        "titulo": "Concentração de Fornecedor",
                        "texto": (
                            f"**{nome}** representa **{pct:.1f}%** (R$ {val/1e6:.2f} Mi) do total histórico. "
                            "Alta dependência cria risco operacional e reduz poder de negociação." + ctx
                        )
                    })

        # CC com maior alta e maior queda
        if not df_r_ant.empty:
            cc_a = df_r.groupby('Desc centro custo')['Valor aprop fin'].sum()
            cc_b = df_r_ant.groupby('Desc centro custo')['Valor aprop fin'].sum()
            cc_m = pd.DataFrame({'a': cc_a, 'b': cc_b}).fillna(0)
            cc_m['d'] = cc_m.apply(lambda r: delta_pct(r['a'], r['b']), axis=1)
            cc_m = cc_m.dropna(subset=['d'])
            if not cc_m.empty:
                cc_alta  = cc_m['d'].idxmax()
                d_alta   = cc_m.loc[cc_alta, 'd']
                v_alta   = cc_m.loc[cc_alta, 'a']
                cc_queda = cc_m['d'].idxmin()
                d_queda  = cc_m.loc[cc_queda, 'd']
                v_queda  = cc_m.loc[cc_queda, 'a']
                if d_alta > 10:
                    insights.append({"tipo": "alerta" if d_alta > 30 else "estrategia",
                        "icone": "🔺",
                        "titulo": f"Maior Alta: {cc_alta[:45]}",
                        "texto": (
                            f"Desembolso de **R$ {v_alta:,.0f}** — alta de **{d_alta:+.1f}%**. "
                            + ("Verificar aderência ao orçamento aprovado." if d_alta > 30
                               else "Tendência de crescimento a monitorar.")
                        )
                    })
                if d_queda < -10:
                    insights.append({"tipo": "sucesso", "icone": "🔻",
                        "titulo": f"Maior Queda: {cc_queda[:45]}",
                        "texto": (
                            f"Desembolso de **R$ {v_queda:,.0f}** — queda de **{d_queda:.1f}%**. "
                            "Pode indicar eficiência, redução de atividade ou pagamentos concentrados no período anterior."
                        )
                    })

        # Juros e descontos
        juros  = (df_p['Valor juros'].sum() + df_p.get('Valor multa', pd.Series([0])).sum()
                  if 'Valor juros' in df_p.columns else 0)
        descs  = df_p['Valor desconto'].sum() if 'Valor desconto' in df_p.columns else 0
        if descs > juros and descs > 0:
            insights.append({"tipo": "sucesso", "icone": "🏆",
                "titulo": "Eficiência em Contas a Pagar",
                "texto": (
                    f"R$ **{descs:,.2f}** capturados em descontos superam a fuga com juros/multas "
                    f"(R$ {juros:,.2f}). Sinal positivo de gestão de prazo."
                )
            })
        elif total_hist and juros > total_hist * 0.005:
            insights.append({"tipo": "perigo", "icone": "🚨",
                "titulo": "Fuga de Capital — Juros e Multas",
                "texto": (
                    f"**R$ {juros:,.2f}** perdidos em juros/multas — "
                    f"{juros/total_hist*100:.2f}% do total pago. "
                    "Priorize antecipação dos títulos com maior exposição."
                )
            })

        # Previsão
        if not projecao.empty:
            media_f  = projecao.mean()
            mes_pico = projecao.idxmax()
            val_pico = projecao.max()
            ritmo    = vol_atual / max(n_dias / 30, 1)
            d_prev   = delta_pct(media_f, ritmo)
            insights.append({"tipo": "estrategia", "icone": "💼",
                "titulo": "Diretriz de Liquidez — Próximo Trimestre",
                "texto": (
                    f"Projeção de **R$ {media_f/1e6:.2f} Mi/mês** nos próximos 3 meses. "
                    f"Pico esperado em **{mes_pico.strftime('%B/%Y')}** (R$ {val_pico/1e6:.2f} Mi). "
                    + (f"Isso representa **{d_prev:+.1f}%** em relação ao ritmo atual." if d_prev else "")
                )
            })

        return insights


# ==========================================
# 6. COMPONENTES VISUAIS
# ==========================================
def card_html(titulo: str, valor: str, delta_txt: str, delta_cor: str) -> str:
    return f"""<div class="metric-card">
        <div class="metric-title">{titulo}</div>
        <div class="metric-value">{valor}</div>
        <div class="metric-delta" style="color:{delta_cor};">{delta_txt}</div>
    </div>"""


def fig_barras_comparativo(
    df_atual: pd.DataFrame, df_ant: pd.DataFrame,
    col_grupo: str, col_val: str, titulo: str, cor_atual: str, n: int = 8
) -> go.Figure:
    g_atual = df_atual.groupby(col_grupo)[col_val].sum().nlargest(n)
    g_ant   = df_ant.groupby(col_grupo)[col_val].sum().reindex(g_atual.index).fillna(0)
    labels  = [str(c)[:32] for c in g_atual.index]

    deltas, cores_d = [], []
    for a, b in zip(g_atual.values, g_ant.values):
        d = delta_pct(a, b)
        deltas.append(f"{d:+.0f}%" if d is not None else "—")
        cores_d.append("#ef4444" if (d or 0) > 0 else "#22c55e")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Período anterior', y=labels, x=g_ant.values,
        orientation='h', marker_color='#e2e8f0',
        hovertemplate='<b>%{y}</b><br>Anterior: R$ %{x:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Período atual', y=labels, x=g_atual.values,
        orientation='h', marker_color=cor_atual,
        text=deltas, textposition='outside',
        textfont=dict(color=cores_d, size=11),
        hovertemplate='<b>%{y}</b><br>Atual: R$ %{x:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        barmode='overlay', title=titulo,
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400, margin=dict(r=70)
    )
    return fig


def fig_tendencia(df: pd.DataFrame, col: str) -> go.Figure:
    ts = df.set_index('Data pagamento').resample('MS')[col].sum().reset_index()
    ts.columns = ['Mes', 'Valor']
    ts['Label'] = ts['Mes'].dt.strftime('%b/%y')
    media = ts['Valor'].mean()
    std   = ts['Valor'].std()
    ts['Atipico'] = ts['Valor'] > media + std

    fig = go.Figure()
    fig.add_hline(y=media, line_dash='dot', line_color='#94a3b8',
                  annotation_text=f'Média R$ {media/1e6:.2f}Mi', annotation_position='top left')
    fig.add_trace(go.Scatter(
        x=ts['Label'], y=ts['Valor'], mode='lines+markers',
        name='Mensal', line=dict(color='#1e293b', width=2.5),
        marker=dict(
            color=ts['Atipico'].map({True: '#d36b32', False: '#1e293b'}),
            size=ts['Atipico'].map({True: 10, False: 6}),
        ),
        hovertemplate='<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>'
    ))
    for _, row in ts[ts['Atipico']].iterrows():
        fig.add_annotation(
            x=row['Label'], y=row['Valor'],
            text=f"R$ {row['Valor']/1e6:.1f}Mi",
            showarrow=True, arrowhead=2, ax=0, ay=-32,
            font=dict(size=11, color='#d36b32')
        )
    fig.update_layout(
        title='Tendência Mensal — Histórico Completo (meses acima da média destacados)',
        yaxis_title='R$', hovermode='x unified'
    )
    return fig


# ==========================================
# 7. APP
# ==========================================
st.set_page_config(
    page_title="Dashboard Financeiro | Contas Pagas",
    page_icon="💰", layout="wide"
)

st.markdown("""<style>
.metric-card {
    border-left: 4px solid #d36b32; padding: 14px 18px;
    border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.07); min-height: 92px;
}
.metric-value { font-size: 22px; font-weight: 700; color: #1e293b; margin: 4px 0 2px; }
.metric-title { font-size: 11px; color: #6d8ca0; text-transform: uppercase; font-weight: 600; letter-spacing: .5px; }
.metric-delta { font-size: 12px; font-weight: 600; margin-top: 3px; }
.insight-card { padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; }
.insight-alerta     { background:#fffbeb; border-color:#f59e0b; }
.insight-sucesso    { background:#f0fdf4; border-color:#22c55e; }
.insight-perigo     { background:#fef2f2; border-color:#ef4444; }
.insight-estrategia { background:#f0f9ff; border-color:#0ea5e9; }
.insight-titulo { font-weight:700; font-size:15px; margin-bottom:5px; }
.banner {
    background: linear-gradient(90deg,#1e293b 0%,#334155 100%);
    color:#f8fafc; border-radius:8px; padding:12px 20px;
    margin-bottom:16px; display:flex; gap:32px; align-items:center;
}
.banner-label { font-weight:700; font-size:15px; color:#f1f5f9; }
.banner-sub   { color:#94a3b8; font-size:12px; margin-bottom:2px; }
</style>""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Pipeline Analítico")
arquivo = st.sidebar.file_uploader("Upload do Extrato (CSV Sienge)", type=['csv'])

if not arquivo:
    st.info("📂 Faça o upload do arquivo CSV exportado do ERP Sienge para iniciar.")
    st.stop()

try:
    with st.spinner("Processando dados..."):
        df_rateios, df_pagamentos = DataPipeline.extrair_e_limpar(arquivo)
except ValueError as e:
    st.error(f"🚫 {e}")
    st.stop()

if df_rateios.empty:
    st.error("Arquivo sem linhas válidas com Valor aprop fin > 0.")
    st.stop()

min_d = df_rateios['Data pagamento'].min().date()
max_d = df_rateios['Data pagamento'].max().date()
def_start = max(min_d, max_d - datetime.timedelta(days=90))

st.sidebar.markdown("### 📅 Período de Análise")
st.sidebar.caption("O período anterior equivalente é calculado automaticamente.")
filtro = st.sidebar.date_input(
    "Intervalo", value=(def_start, max_d),
    min_value=min_d, max_value=max_d
)

st.sidebar.markdown("### 🔍 Filtros")
centros_sel = st.sidebar.multiselect(
    "Centro de Custo",
    options=sorted(df_rateios['Desc centro custo'].dropna().unique()),
    default=[], placeholder="Todos"
)
planos_sel = st.sidebar.multiselect(
    "Plano Financeiro",
    options=sorted(df_rateios['Desc plano fin'].dropna().unique()),
    default=[], placeholder="Todos"
)

# ---- DATAS ----
if isinstance(filtro, tuple) and len(filtro) == 2:
    d_ini, d_fim = filtro
else:
    d_ini, d_fim = def_start, max_d

n_dias = (d_fim - d_ini).days + 1

# ---- FILTRAR ATUAL ----
def filtrar(df):
    mask = (df['Data pagamento'].dt.date >= d_ini) & (df['Data pagamento'].dt.date <= d_fim)
    r = df.loc[mask].copy()
    if centros_sel and 'Desc centro custo' in r.columns:
        r = r[r['Desc centro custo'].isin(centros_sel)]
    if planos_sel and 'Desc plano fin' in r.columns:
        r = r[r['Desc plano fin'].isin(planos_sel)]
    return r

df_dash   = filtrar(df_rateios)
df_p_dash = filtrar(df_pagamentos)

if df_dash.empty:
    st.warning("⚠️ Nenhum dado no período/filtros selecionados.")
    st.stop()

# ---- PERÍODO ANTERIOR ----
df_r_ant_full, ant_ini, ant_fim = periodo_anterior(df_rateios, d_ini, d_fim)
df_p_ant_full, _,       _       = periodo_anterior(df_pagamentos, d_ini, d_fim)

def filtrar_ant(df):
    r = df.copy()
    if centros_sel and 'Desc centro custo' in r.columns:
        r = r[r['Desc centro custo'].isin(centros_sel)]
    if planos_sel and 'Desc plano fin' in r.columns:
        r = r[r['Desc plano fin'].isin(planos_sel)]
    return r

df_r_ant = filtrar_ant(df_r_ant_full)
df_p_ant = filtrar_ant(df_p_ant_full)

# ---- MÉTRICAS ----
v_at = df_dash['Valor aprop fin'].sum()
v_an = df_r_ant['Valor aprop fin'].sum()

chave = '_chave' if '_chave' in df_dash.columns else None
np_at = df_dash.drop_duplicates(subset=[chave] if chave else None).shape[0]
np_an = df_r_ant.drop_duplicates(subset=[chave] if chave else None).shape[0]

cr_at = df_dash['Nome credor'].nunique()
cr_an = df_r_ant['Nome credor'].nunique()

tk_at = v_at / max(np_at, 1)
tk_an = v_an / max(np_an, 1)

d_v  = delta_pct(v_at, v_an)
d_np = delta_pct(np_at, np_an)
d_cr = delta_pct(cr_at, cr_an)
d_tk = delta_pct(tk_at, tk_an)

# ==========================================
# LAYOUT
# ==========================================
import os

col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=80)
    elif os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=80)
with col_title:
    st.title("💰 Gestão Executiva de Desembolso")

st.markdown(f"""<div class="banner">
    <div><div class="banner-sub">Período analisado</div>
         <div class="banner-label">{d_ini.strftime('%d/%m/%Y')} → {d_fim.strftime('%d/%m/%Y')} ({n_dias} dias)</div></div>
    <div><div class="banner-sub">Comparando com</div>
         <div class="banner-label">{ant_ini.strftime('%d/%m/%Y')} → {ant_fim.strftime('%d/%m/%Y')}</div></div>
    <div><div class="banner-sub">Histórico completo</div>
         <div class="banner-label">{min_d.strftime('%d/%m/%Y')} → {max_d.strftime('%d/%m/%Y')}</div></div>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(card_html("Volume no Período", f"R$ {v_at/1e6:.2f} Mi",
    fmt_delta(d_v), cor_delta(d_v)), unsafe_allow_html=True)
c2.markdown(card_html("Pagamentos", f"{np_at:,}",
    fmt_delta(d_np), cor_delta(d_np)), unsafe_allow_html=True)
c3.markdown(card_html("Fornecedores Únicos", str(cr_at),
    fmt_delta(d_cr), cor_delta(d_cr)), unsafe_allow_html=True)
c4.markdown(card_html("Ticket Médio", f"R$ {tk_at:,.0f}",
    fmt_delta(d_tk), cor_delta(d_tk)), unsafe_allow_html=True)

st.write("---")

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Comparativo de Períodos",
    "📈 Tendência Histórica",
    "🧠 Inteligência Analítica",
    "📋 Dados Brutos"
])

# ---- ABA 1 ----
with aba1:
    tem_ant = not df_r_ant.empty

    if not tem_ant:
        st.info("⚠️ Não há dados no período anterior — exibindo apenas o período atual.")

    c1g, c2g = st.columns(2)
    with c1g:
        if tem_ant:
            st.plotly_chart(fig_barras_comparativo(
                df_dash, df_r_ant, 'Desc centro custo', 'Valor aprop fin',
                'Centro de Custo — Atual vs. Anterior', '#1e293b'
            ), use_container_width=True)
        else:
            cc = df_dash.groupby('Desc centro custo')['Valor aprop fin'].sum().nlargest(8).reset_index()
            f  = px.bar(cc, x='Valor aprop fin', y='Desc centro custo', orientation='h',
                        title='Top Centros de Custo', labels={'Valor aprop fin': 'R$', 'Desc centro custo': ''})
            f.update_traces(marker_color='#1e293b')
            f.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(f, use_container_width=True)

    with c2g:
        if tem_ant:
            st.plotly_chart(fig_barras_comparativo(
                df_dash, df_r_ant, 'Desc plano fin', 'Valor aprop fin',
                'Plano Financeiro — Atual vs. Anterior (Δ% anotado)', '#d36b32'
            ), use_container_width=True)
        else:
            pf = df_dash.groupby('Desc plano fin')['Valor aprop fin'].sum().nlargest(8).reset_index()
            f  = px.bar(pf, x='Valor aprop fin', y='Desc plano fin', orientation='h',
                        title='Top Planos Financeiros', labels={'Valor aprop fin': 'R$', 'Desc plano fin': ''})
            f.update_traces(marker_color='#d36b32')
            f.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(f, use_container_width=True)

    if tem_ant:
        st.write("---")
        st.subheader("Tabela de Variação por Centro de Custo")
        cc_a = df_dash.groupby('Desc centro custo')['Valor aprop fin'].sum().rename('Atual')
        cc_b = df_r_ant.groupby('Desc centro custo')['Valor aprop fin'].sum().rename('Anterior')
        tab  = pd.DataFrame([cc_a, cc_b]).T.fillna(0).sort_values('Atual', ascending=False).head(15)
        tab['Δ R$'] = tab['Atual'] - tab['Anterior']
        tab['Δ %']  = tab.apply(lambda r: delta_pct(r['Atual'], r['Anterior']), axis=1)
        tab['Atual']    = tab['Atual'].map('R$ {:,.0f}'.format)
        tab['Anterior'] = tab['Anterior'].map('R$ {:,.0f}'.format)
        tab['Δ R$']     = tab['Δ R$'].map('R$ {:+,.0f}'.format)
        tab['Δ %']      = tab['Δ %'].apply(lambda x: f"{x:+.1f}%" if x is not None else "—")
        st.dataframe(tab, use_container_width=True)

# ---- ABA 2 ----
with aba2:
    st.plotly_chart(fig_tendencia(df_rateios, 'Valor aprop fin'), use_container_width=True)

    st.write("---")
    evol = df_dash.groupby('Data pagamento')['Valor aprop fin'].sum().reset_index()
    fig_a = px.area(evol, x='Data pagamento', y='Valor aprop fin',
                    title='Evolução Diária — Período Selecionado',
                    labels={'Valor aprop fin': 'R$', 'Data pagamento': ''})
    fig_a.update_traces(line_color='#1e293b', fillcolor='rgba(30,41,59,0.08)')
    fig_a.update_layout(hovermode='x unified')
    st.plotly_chart(fig_a, use_container_width=True)

    top6 = df_rateios.groupby('Desc centro custo')['Valor aprop fin'].sum().nlargest(6).index.tolist()
    df_s = df_rateios[df_rateios['Desc centro custo'].isin(top6)].copy()
    df_s['Mês'] = df_s['Data pagamento'].dt.to_period('M').dt.to_timestamp()
    men  = df_s.groupby(['Mês', 'Desc centro custo'])['Valor aprop fin'].sum().reset_index()
    fig_s = px.bar(men, x='Mês', y='Valor aprop fin', color='Desc centro custo',
                   title='Evolução Mensal por CC (Top 6)', barmode='stack',
                   labels={'Valor aprop fin': 'R$', 'Mês': '', 'Desc centro custo': 'CC'})
    fig_s.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_s, use_container_width=True)

# ---- ABA 3 ----
with aba3:
    st.subheader("Narrativa Executiva e Predição Estatística")
    st.caption("Análise contextualizada do período filtrado vs. anterior. Previsão usa o histórico completo.")

    col_t, col_g = st.columns([1, 1])

    try:
        hist, proj = ForecastingEngine.prever(df_rateios, 'Valor aprop fin')

        with col_t:
            ins = GeradorDeInsights.analisar(
                df_dash, df_p_dash,
                df_r_ant, df_p_ant,
                proj, d_ini, d_fim
            )
            for it in ins:
                st.markdown(f"""<div class="insight-card insight-{it['tipo']}">
                    <div class="insight-titulo">{it['icone']} {it['titulo']}</div>
                    <div style="font-size:14px;line-height:1.6;">{it['texto']}</div>
                </div>""", unsafe_allow_html=True)

        with col_g:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=hist.index, y=hist.values, name='Realizado (12m)',
                line=dict(color='#1e293b', width=3),
                hovertemplate='<b>%{x|%b/%Y}</b><br>R$ %{y:,.0f}<extra></extra>'
            ))
            fig3.add_trace(go.Scatter(
                x=proj.index, y=proj.values, name='Projeção (3m)',
                line=dict(color='#d36b32', width=3, dash='dash'),
                fill='tozeroy', fillcolor='rgba(211,107,50,0.07)',
                hovertemplate='<b>%{x|%b/%Y}</b><br>R$ %{y:,.0f}<extra></extra>'
            ))
            fig3.update_layout(
                title='Forecasting de Saída de Caixa',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig3, use_container_width=True)

    except ValueError as e:
        st.warning(str(e))

    st.write("---")
    df_tree = (df_rateios.groupby(['Desc centro custo', 'Desc plano fin'])['Valor aprop fin']
               .sum().reset_index().nlargest(50, 'Valor aprop fin'))
    if not df_tree.empty:
        fig_t = px.treemap(df_tree, path=['Desc centro custo', 'Desc plano fin'],
                           values='Valor aprop fin',
                           title='Mapa de Gastos — CC → Plano Financeiro (Histórico)',
                           color='Valor aprop fin', color_continuous_scale='Blues')
        fig_t.update_layout(height=450)
        st.plotly_chart(fig_t, use_container_width=True)

# ---- ABA 4 ----
with aba4:
    st.subheader("Dados Brutos — Período Filtrado")
    cols = [c for c in ['Data pagamento', 'Nome credor', 'Desc centro custo',
                         'Desc plano fin', 'Valor aprop fin', 'Valor Líquido',
                         'Desc forma pagto'] if c in df_dash.columns]
    df_ex = df_dash[cols].copy()
    df_ex['Data pagamento'] = df_ex['Data pagamento'].dt.strftime('%d/%m/%Y')
    st.dataframe(df_ex, use_container_width=True, height=420)
    st.download_button("⬇️ Exportar CSV filtrado",
        data=df_ex.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
        file_name=f"contas_pagas_{d_ini}_{d_fim}.csv", mime='text/csv')

st.sidebar.markdown("---")
st.sidebar.caption(
    f"🗂️ Período atual: {len(df_dash):,} linhas\n"
    f"📦 Período anterior: {ant_ini.strftime('%d/%m/%y')} → {ant_fim.strftime('%d/%m/%y')}\n"
    f"🏢 CCs distintos: {df_dash['Desc centro custo'].nunique()}"
)
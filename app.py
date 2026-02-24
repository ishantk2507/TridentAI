"""
Streamlit UI for TridentAI: Adaptive Multi-Regime Stock Trading Agent.

Real-time dashboard showing:
- Live stock prices and sentiment
- Time-series forecasts (Prophet + Attention LSTM)
- RL agent trading signals
- Backtest comparisons
- Risk metrics (Sharpe, Sortino, Max Drawdown)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src import (
    StockProfiler, TridentForecaster, SentimentEngine,
    load_or_train_agent, TridentTradingEnv, set_seeds
)

# Seeding for reproducibility
set_seeds(42)

# Streamlit page config
st.set_page_config(
    page_title="Trident AI Platform",
    layout="wide",
    page_icon="🔱",
    initial_sidebar_state="expanded"
)

# Cache sentiment engine
@st.cache_resource
def get_sentiment_engine():
    return SentimentEngine()

sent_engine = get_sentiment_engine()

# ==========================================
# GLOBAL MARKET CONFIGURATION
# ==========================================
MARKET_CONFIG = {
    "🇺🇸 USA (NYSE/NASDAQ)": {
        "benchmark": "SPY",
        "currency": "$",
        "suffix": "",
        "default_ticker": "NVDA",
        "presets": {
            "Chaos (Extreme Vol)": ["MSTR", "COIN", "MARA"],
            "Tech (High Vol)": ["NVDA", "TSLA", "AMD", "PLTR"],
            "Stable (Med Vol)": ["AAPL", "MSFT", "GOOGL"],
            "Defensive (Low Vol)": ["KO", "JNJ", "PG", "WMT"]
        }
    },
    "🇮🇳 India (NSE)": {
        "benchmark": "^NSEI",
        "currency": "₹",
        "suffix": ".NS",
        "default_ticker": "ADANIENT.NS",
        "presets": {
            "Chaos (Extreme Vol)": ["ADANIENT.NS", "YESBANK.NS", "IDEA.NS"],
            "Growth (High Vol)": ["TATAELXSI.NS", "BAJFINANCE.NS", "SUZLON.NS"],
            "Blue Chip (Med Vol)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
            "Defensive (Low Vol)": ["ITC.NS", "HINDUNILVR.NS", "ASIANPAINT.NS"]
        }
    }
}


def analyze_stock_pipeline(
    ticker: str,
    start_date,
    end_date,
    benchmark_ticker: str,
    split_date=None,
    force_retrain: bool = False
):
    """Main analysis pipeline with Auto-Training Logic"""
    
    # STEP 0: Download data
    with st.spinner(f"📊 Fetching Data..."):
        try:
            gen_df_full = yf.download(
                benchmark_ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=True
            )
            spec_df_full = yf.download(
                ticker, start=start_date, end=end_date,
                progress=False, auto_adjust=True
            )
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            return None
        
        if len(gen_df_full) == 0 or len(spec_df_full) == 0:
            st.error(f"❌ No data for {ticker}.")
            return None
        
        # --- SPLIT LOGIC ---
        if split_date:
            split_ts = pd.to_datetime(split_date)
            gen_train = gen_df_full[gen_df_full.index < split_ts]
            spec_train = spec_df_full[spec_df_full.index < split_ts]
            gen_test = gen_df_full[gen_df_full.index >= split_ts]
            spec_test = spec_df_full[spec_df_full.index >= split_ts]
            
            if len(spec_test) < 50:
                st.warning(f"⚠️ Not enough test data after {split_date}. Using full dataset.")
                gen_train, spec_train = gen_df_full, spec_df_full
                gen_test, spec_test = gen_df_full, spec_df_full
                split_mode = False
            else:
                split_mode = True
        else:
            gen_train, spec_train = gen_df_full, spec_df_full
            gen_test, spec_test = gen_df_full, spec_df_full
            split_mode = False
        
        # Profile the stock
        profile = StockProfiler.analyze(spec_train, ticker)
        hyperparams = StockProfiler.get_hyperparams(spec_train, ticker)
        
        with st.sidebar:
            st.divider()
            st.markdown(f"**{ticker} Profile**")
            st.caption(f"Regime: {profile['regime']}")
            st.caption(f"Annual Vol: {profile['annual_vol']:.1f}%")
            if split_mode:
                st.success(f"🧪 Research Mode: Training until {split_date}")
            else:
                st.info("🚀 Production Mode: Using/Training Full History")
    
    # STEP 1: Sentiment
    with st.spinner(f"🧠 Analyzing News..."):
        try:
            sentiment_score, headlines = sent_engine.get_sentiment(ticker)
        except Exception:
            sentiment_score, headlines = 0.0, ["Unavailable"]
    
    # STEP 2: Forecasting
    with st.spinner(f"📈 Training Forecaster..."):
        forecaster = TridentForecaster(
            ticker,
            general_ticker=benchmark_ticker,
            hyperparams=hyperparams
        )
        
        # Always train Forecaster (Fast & Essential)
        success = forecaster.train(gen_train, spec_train)
        if not success:
            st.error("❌ Training failed - insufficient history")
            return None
        
        gen_sig, spec_sig, prophet_sig = forecaster.batch_predict(gen_test, spec_test)
        
        if split_mode:
            gen_sig_train, spec_sig_train, prophet_sig_train = forecaster.batch_predict(gen_train, spec_train)
        else:
            gen_sig_train, spec_sig_train, prophet_sig_train = gen_sig, spec_sig, prophet_sig
    
    # STEP 3: Adaptive RL Agent (Smart Loading)
    with st.spinner(f"🤖 Managing AI Agent..."):
        model, status = load_or_train_agent(
            ticker,
            spec_train,
            gen_train,
            (gen_sig_train, spec_sig_train, prophet_sig_train),
            hyperparams,
            sentiment_score=sentiment_score,
            retrain=force_retrain
        )
        
        if status == "Trained":
            st.toast(f"🆕 Trained new agent for {ticker}!")
        else:
            st.toast(f"✅ Loaded existing agent for {ticker}")
        
        env = TridentTradingEnv(
            spec_test,
            gen_test,
            (gen_sig, spec_sig, prophet_sig),
            sentiment_score=sentiment_score,
            hyperparams=hyperparams,
            training_mode=False
        )
    
    # STEP 4: Prediction
    with st.spinner(f"🔮 Generating Live Prediction..."):
        try:
            recent = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
            from src.utils import extract_close_price
            close_series = extract_close_price(recent, ticker)
            close_data = close_series.values
            
            if len(close_data) < forecaster.lookback + 1:
                st.error(f"Not enough recent data. Need {forecaster.lookback+1} points.")
                return None
            
            live_slice = close_data[-(forecaster.lookback + 1):]
            current_price = float(close_data[-1])
            prev_price = float(close_data[-2])
            daily_return = ((current_price - prev_price) / prev_price) * 100
            price_change = current_price - prev_price
            
            # Get live prediction from agent
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            target_alloc = float(action[0])
            
            return {
                'ticker': ticker,
                'profile': profile,
                'hyperparams': hyperparams,
                'sentiment': sentiment_score,
                'headlines': headlines,
                'forecaster': forecaster,
                'model': model,
                'env': env,
                'current_price': current_price,
                'daily_return': daily_return,
                'price_change': price_change,
                'target_allocation': target_alloc,
                'gen_signal': gen_sig,
                'spec_signal': spec_sig,
                'prophet_signal': prophet_sig,
                'gen_train': gen_train,
                'spec_train': spec_train,
                'gen_test': gen_test,
                'spec_test': spec_test,
                'split_mode': split_mode
            }
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return None


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🔱 Trident AI Platform")
st.markdown("*Adaptive Multi-Regime Stock Trading Agent*")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    market_choice = st.radio("Select Market:", list(MARKET_CONFIG.keys()))
    market = MARKET_CONFIG[market_choice]
    
    preset_choice = st.selectbox(
        "Stock Preset:",
        list(market['presets'].keys())
    )
    
    preset_tickers = market['presets'][preset_choice]
    
    selected_ticker = st.selectbox(
        "Or Enter Custom Ticker:",
        preset_tickers + ["---Custom---"],
        index=0
    )
    
    if selected_ticker == "---Custom---":
        ticker = st.text_input("Enter Ticker:", value="NVDA")
    else:
        ticker = selected_ticker
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=pd.Timestamp('2023-01-01')
        )
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=pd.Timestamp.now()
        )
    
    st.divider()
    
    split_mode = st.checkbox("🧪 Research Mode (Train/Test Split)?", value=False)
    if split_mode:
        split_date = st.date_input(
            "Split Date (Training until):",
            value=pd.Timestamp.now() - pd.Timedelta(days=90)
        )
    else:
        split_date = None
    
    force_retrain = st.checkbox("🔄 Force Retrain Agent?", value=False)

# Main content
if st.sidebar.button("▶️ Analyze Stock", use_container_width=True, type="primary"):
    
    results = analyze_stock_pipeline(
        ticker=ticker.replace(".NS", "") if ticker != "---Custom---" else ticker,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=market['benchmark'],
        split_date=split_date,
        force_retrain=force_retrain
    )
    
    if results:
        # Store in session for visualization
        st.session_state.results = results
        st.success("✅ Analysis Complete!")
    else:
        st.error("❌ Analysis Failed")

# Display results if available
if 'results' in st.session_state:
    r = st.session_state.results
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Price", f"${r['current_price']:.2f}", f"{r['price_change']:+.2f}")
    with col2:
        st.metric("📊 Daily Return", f"{r['daily_return']:.2f}%")
    with col3:
        st.metric("🧠 Sentiment", f"{r['sentiment']:.2f}", "Bullish" if r['sentiment'] > 0 else "Bearish")
    with col4:
        st.metric("🤖 Allocation", f"{r['target_allocation']*100:.1f}%", "Long" if r['target_allocation'] > 0.5 else "Conservative")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signals", "📈 Forecasts", "📰 News", "ℹ️ Details"])
    
    with tab1:
        st.markdown("### Trading Signals")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Benchmark Signal", f"{r['gen_signal'][-1]:.3f}")
        with col2:
            st.metric("Stock Signal", f"{r['spec_signal'][-1]:.3f}")
        with col3:
            st.metric("Prophet Signal", f"{r['prophet_signal'][-1]:.3f}")
    
    with tab2:
        st.markdown("### Time-Series Forecasts")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                y=r['spec_test']['Close'].values,
                name='Close Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Signals
        fig.add_trace(
            go.Scatter(
                y=r['spec_signal'],
                name='Stock Signal',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Latest Financial News")
        if r['headlines'] and r['headlines'][0] != "No Recent News Found":
            for i, headline in enumerate(r['headlines'], 1):
                st.write(f"{i}. {headline}")
        else:
            st.info("No recent news found")
    
    with tab4:
        st.markdown("### System Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Stock Profile**")
            st.write(f"- Regime: {r['profile']['regime']}")
            st.write(f"- Annual Vol: {r['profile']['annual_vol']:.1f}%")
            st.write(f"- Trend Strength: {r['profile']['trend_strength']:.2f}")
        with col2:
            st.write("**Hyperparameters**")
            st.write(f"- Hidden Gen: {r['hyperparams']['hidden_gen']}")
            st.write(f"- Hidden Spec: {r['hyperparams']['hidden_spec']}")
            st.write(f"- Reward Scale: {r['hyperparams']['reward_scale']}")
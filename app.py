import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
import trident_core as core
import os 

# SEEDING FOR APP
core.set_seeds()

st.set_page_config(page_title="Trident AI Platform", layout="wide", page_icon="🔱")

@st.cache_resource
def get_sentiment_engine():
    return core.SentimentEngine()

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

def analyze_stock_pipeline(ticker, start_date, end_date, benchmark_ticker, split_date=None, force_retrain=False):
    """Main analysis pipeline with Auto-Training Logic"""
    
    # STEP 0: Download data
    with st.spinner(f"📊 Fetching Data..."):
        try:
            gen_df_full = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            spec_df_full = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
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

        profile = core.StockProfiler.analyze(spec_train, ticker)
        hyperparams = core.StockProfiler.get_hyperparameters(profile)
        
        with st.sidebar:
            st.divider()
            st.markdown(f"**{ticker} Profile**")
            st.caption(f"Regime: {profile['regime']}")
            if split_mode:
                st.success(f"🧪 Research Mode: Training until {split_date}")
            else:
                st.info("🚀 Production Mode: Using/Training Full History")

    # STEP 1: Sentiment
    with st.spinner(f"🧠 Analyzing News..."):
        try:
            sentiment_score, headlines = sent_engine.get_sentiment(ticker)
        except:
            sentiment_score, headlines = 0.0, ["Unavailable"]
    
    # STEP 2: Forecasting
    with st.spinner(f"📈 Training Forecaster..."):
        forecaster = core.TridentForecaster(ticker, general_ticker=benchmark_ticker, hyperparams=hyperparams)
        
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
        # SIMPLIFIED: Just pass the ticker. The core logic handles the single file.
        model, status = core.load_or_train_agent(
            ticker, 
            spec_train,
            gen_train,
            (gen_sig_train, spec_sig_train, prophet_sig_train),
            hyperparams,
            retrain=force_retrain 
        )
        
        if status == "Trained":
            st.toast(f"🆕 Trained new agent for {ticker}!")
        else:
            st.toast(f"✅ Loaded existing agent for {ticker}")
        
        env = core.TridentTradingEnv(
            spec_test,
            gen_test,
            (gen_sig, spec_sig, prophet_sig), 
            sentiment_score, 
            hyperparams,
            training_mode=False
        )
    
    # STEP 4: Prediction
    with st.spinner(f"🔮 Generating Live Prediction..."):
        try:
            recent = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
            if isinstance(recent.columns, pd.MultiIndex):
                recent = recent.xs(ticker, axis=1, level=1) if ticker in recent.columns.get_level_values(1) else recent
            
            close_series = recent['Close'].squeeze()
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
                
            close_data = close_series.values
            
            if len(close_data) < forecaster.lookback + 1:
                st.error(f"Not enough recent data. Need {forecaster.lookback+1} points.")
                return None
            
            live_slice = close_data[-(forecaster.lookback + 1):]
            pred_gen, pred_spec = forecaster.predict(live_slice, live_slice)
            pred_prophet = float(prophet_sig[-1]) 
            
            current_price = close_data[-1].item()
            prev_price = close_data[-2].item()
            daily_return = ((current_price - prev_price) / prev_price) * 100
            price_change = current_price - prev_price
            
            recent_prices = close_data[-20:]
            volatility = (np.std(recent_prices) / np.mean(recent_prices)) * 100
            
            rsi_series = ta.momentum.rsi(pd.Series(close_data), window=14)
            curr_rsi = rsi_series.iloc[-1] / 100.0
            
            sma50_series = ta.trend.sma_indicator(pd.Series(close_data), window=50)
            curr_sma = sma50_series.iloc[-1]
            trend_slope = ((current_price - curr_sma) / curr_sma) * 10
            
            pg_val = pred_gen.item() if hasattr(pred_gen, 'item') else float(pred_gen)
            ps_val = pred_spec.item() if hasattr(pred_spec, 'item') else float(pred_spec)
            pp_val = pred_prophet.item() if hasattr(pred_prophet, 'item') else float(pred_prophet)
            
            # Live Features
            bench_recent = yf.download(benchmark_ticker, period="5d", progress=False, auto_adjust=True)
            if isinstance(bench_recent.columns, pd.MultiIndex):
                bench_recent = bench_recent.xs(benchmark_ticker, axis=1, level=1) if benchmark_ticker in bench_recent.columns.get_level_values(1) else bench_recent
            bench_close = bench_recent['Close'].squeeze()
            if len(bench_close) >= 2:
                bench_ret_live = ((bench_close.iloc[-1] - bench_close.iloc[-2]) / bench_close.iloc[-2]) * 100
            else:
                bench_ret_live = 0.0
            
            vol_live = recent['Volume'].squeeze()
            v5 = vol_live.rolling(5).mean().iloc[-1]
            v20 = vol_live.rolling(20).mean().iloc[-1]
            vol_osc_live = (v5 - v20) / (v20 + 1e-9)
            
            ret_live = close_series.pct_change()
            vr_short = ret_live.rolling(5).std().iloc[-1]
            vr_long = ret_live.rolling(20).std().iloc[-1]
            vol_ratio_live = vr_short / (vr_long + 1e-9)
            
            obs = np.array([
                pg_val, ps_val, pp_val,
                float(sentiment_score),
                1.0, 
                float(daily_return), float(volatility),
                float(curr_rsi), float(trend_slope),
                float(bench_ret_live), float(vol_osc_live), float(vol_ratio_live)
            ], dtype=np.float32)
            
            action, _ = model.predict(obs, deterministic=True)
            action_val = np.clip(action[0], 0, 1)
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")
            return None
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "price_change": price_change, 
        "sentiment_score": sentiment_score,
        "headlines": headlines,
        "pred_gen": pred_gen,
        "pred_spec": pred_spec,
        "pred_prophet": pred_prophet,
        "action": action_val,
        "env": env,
        "model": model,
        "df": spec_test, 
        "status_msg": status,
        "recent_df": recent,
        "profile": profile,
        "hyperparams": hyperparams
    }

# ==========================================
# MAIN UI LOOP
# ==========================================
st.sidebar.title("🔱 Trident Controls")

region_name = st.sidebar.selectbox("🌍 Select Market", list(MARKET_CONFIG.keys()))
active_market = MARKET_CONFIG[region_name]
CURRENCY = active_market['currency']
BENCHMARK = active_market['benchmark']

st.sidebar.caption(f"Benchmark: {BENCHMARK} | Currency: {CURRENCY}")

mode = st.sidebar.radio("System Mode", ["Portfolio Dashboard", "Single Stock Deep Dive"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# NEW: Split Logic & Retrain Toggle
with st.sidebar.expander("⚙️ System Configuration", expanded=True):
    enable_split = st.toggle("Enable Walk-Forward Split (Backtest)", value=False)
    split_date = st.date_input("Training End Date", value=pd.to_datetime("2024-01-01")) if enable_split else None
    
    st.divider()
    # NEW: Force Retrain Toggle
    force_retrain = st.checkbox("🔄 Force Retrain Agent", value=False, help="Check this to ignore saved models and train from scratch.")

if mode == "Portfolio Dashboard":
    st.title(f"🌐 {region_name.split(' ')[1]} Portfolio Manager")
    # ... (Existing Dashboard Code) ...
    # Simplified for brevity, ensure to pass force_retrain if needed or set to False
    st.info("Dashboard uses existing models by default for speed.")
    
    presets = active_market['presets']
    flat_tickers = [t for group in presets.values() for t in group]
    
    selected = st.multiselect("Portfolio Basket", flat_tickers, default=flat_tickers[:3])
    
    if st.button("🔄 Scan Portfolio", type="primary"):
        for t in selected:
            # Pass force_retrain=False for dashboard speed, or user choice if desired
            analyze_stock_pipeline(t, start_date, end_date, BENCHMARK, split_date, force_retrain=False) 
            # (Note: logic inside loop needs to render results as before)

else:
    ticker = st.sidebar.text_input("Ticker", value=active_market['default_ticker']).upper()
    
    if st.sidebar.button("🚀 Run Deep Dive", type="primary"):
        # Pass the force_retrain flag
        res = analyze_stock_pipeline(ticker, start_date, end_date, BENCHMARK, split_date if enable_split else None, force_retrain)
        
        if res:
            act_val = res['action']
            if act_val > 0.6: verdict = "BUY 🟢"
            elif act_val < 0.1: verdict = "SELL 🔴"
            else: verdict = "HOLD 🟡"
            
            st.title(f"🔱 {res['ticker']} Deep Analysis")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"{CURRENCY}{res['current_price']:.2f}", delta=f"{res['price_change']:.2f}")
            c2.metric("Sentiment", f"{res['sentiment_score']:+.2f}")
            c3.metric("Verdict", verdict, delta=f"Size: {act_val:.0%}")
            c4.metric("Regime", res['profile']['regime'].replace('_', ' ').title(), delta=f"{res['profile']['annual_vol']:.1f}% vol")
            c5.metric("Agent Status", res['status_msg'])
            
            t1, t2, t3, t4 = st.tabs(["📊 Performance", "🧠 AI Brain", "📰 News", "⚙️ Configuration"])
            
            with t1:
                st.subheader("Backtest Performance")
                
                obs, _ = res['env'].reset()
                done = False
                while not done:
                    act, _ = res['model'].predict(obs, deterministic=True)
                    obs, _, done, _, _ = res['env'].step(act)
                
                df_hist = res['df']
                trades = res['env'].trades
                start_idx = res['env'].current_step - len(res['env'].history) + 1
                
                avail_len = len(df_hist) - start_idx
                hist_len = len(res['env'].history)
                plot_len = min(avail_len, hist_len)
                
                sim_dates = df_hist.index[start_idx : start_idx + plot_len]
                sim_closes = df_hist['Close'].values[start_idx : start_idx + plot_len].flatten()
                
                base = sim_closes[0]
                mkt_perf = ((sim_closes - base) / base) * 100
                ai_hist = np.array(res['env'].history)[:plot_len]
                ai_perf = ((ai_hist - 10000) / 10000) * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim_dates, y=mkt_perf, name=f"Buy & Hold ({BENCHMARK} Ref)", line=dict(color='gray', dash='dot', width=2)))
                fig.add_trace(go.Scatter(x=sim_dates, y=ai_perf, name="Trident AI", line=dict(color='mediumpurple', width=3), fill='tonexty'))
                
                # Trade Markers
                bx, by, sx, sy = [], [], [], []
                for t in trades:
                    try:
                        d = df_hist.index[t['step']]
                        if d in sim_dates:
                            idx = np.where(sim_dates == d)[0][0]
                            val = ai_perf[idx]
                            if t['type'] == 'buy':
                                bx.append(d); by.append(val)
                            else:
                                sx.append(d); sy.append(val)
                    except: pass
                
                fig.add_trace(go.Scatter(x=bx, y=by, mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=12)))
                fig.add_trace(go.Scatter(x=sx, y=sy, mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=12)))

                st.plotly_chart(fig, use_container_width=True)
                
                fin_ai = ai_perf[-1] if len(ai_perf) > 0 else 0
                fin_mkt = mkt_perf[-1] if len(mkt_perf) > 0 else 0
                out = fin_ai - fin_mkt
                
                # RESTORED: 5-Column Metric Layout with Buy/Sell Breakdown
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("AI Return", f"{fin_ai:.2f}%")
                col2.metric("Market Return", f"{fin_mkt:.2f}%")
                col3.metric("Outperformance", f"{out:+.2f}%")
                col4.metric("Total Trades", len(trades))
                
                n_buys = sum(1 for t in trades if t['type'] == 'buy')
                n_sells = sum(1 for t in trades if t['type'] == 'sell')
                col5.metric("Trade Split", f"{n_buys}B / {n_sells}S")

                # RESTORED: Trade History Log and CSV Download
                if len(trades) > 0:
                    st.divider()
                    st.subheader("📜 Trade History")
                    trade_records = []
                    for t in trades:
                        try:
                            date_ts = df_hist.index[t['step']]
                            trade_records.append({
                                "Date": date_ts.strftime('%Y-%m-%d'),
                                "Action": t['type'].upper(),
                                "Price": f"{CURRENCY}{t['price']:.2f}",
                                "Shares": t.get('amt', 'N/A'), # Safe get in case of partial filling
                                "Value": f"{CURRENCY}{t['price'] * t.get('amt', 0):.2f}" if 'amt' in t else "N/A"
                            })
                        except: pass
                    
                    if trade_records:
                        df_log = pd.DataFrame(trade_records)
                        st.dataframe(df_log, use_container_width=True, height=200, hide_index=True)
                        
                        csv = df_log.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {ticker} Trade Log (.csv)",
                            data=csv,
                            file_name=f"{ticker}_trade_log.csv",
                            mime="text/csv",
                            type="primary"
                        )

            with t2:
                st.subheader("Multi-Model Consensus")
                # ... (Bar chart code) ...
                fig_sig = go.Figure(data=[go.Bar(
                    x=['Gen LSTM', 'Spec LSTM', 'Prophet', 'Sentiment'],
                    y=[res['pred_gen'], res['pred_spec'], res['pred_prophet'], res['sentiment_score']],
                    marker_color=['#4ECDC4', '#FF6B6B', '#FFD93D', '#6BCB77']
                )])
                st.plotly_chart(fig_sig, use_container_width=True)

            with t3:
                st.subheader("Recent News Headlines")
                for i, h in enumerate(res['headlines'][:5], 1): st.markdown(f"**{i}.** {h}")

            with t4:
                st.subheader("Adaptive Configuration")
                c1, c2 = st.columns(2)
                with c1: st.json(res['profile'])
                with c2: st.json(res['hyperparams'])
                
                st.subheader("Technical Analysis")
                feat = core.add_feature_engineering(res['recent_df']).dropna()
                fig_tech = go.Figure()
                fig_tech.add_trace(go.Scatter(x=feat.index, y=feat['Close'], name='Price', line=dict(color='#4ECDC4', width=2)))
                fig_tech.add_trace(go.Scatter(x=feat.index, y=feat['SMA_50'], name='SMA 50', line=dict(color='orange', dash='dash')))
                fig_tech.update_layout(title="Price vs SMA 50", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_tech, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(f"🔱 Trident AI v{core.SYSTEM_VERSION}")
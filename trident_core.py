import os
import shutil
import random
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler
import ta
from prophet import Prophet
import logging

# Silence Prophet logs
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# ==========================================
# SYSTEM VERSIONING
# ==========================================
SYSTEM_VERSION = "v20_single_model_architecture"

# ==========================================
# SEEDING
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds()

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "v3/trident_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# STOCK PROFILE ANALYZER (6-TIER SYSTEM)
# ==========================================
class StockProfiler:
    """Analyzes stock characteristics to determine optimal hyperparameters"""
    
    @staticmethod
    def analyze(df, ticker):
        # Robust MultiIndex handling
        if isinstance(df.columns, pd.MultiIndex):
            try:
                c = df.xs(ticker, level=1, axis=1)['Close']
            except KeyError:
                c = df['Close']
        else:
            c = df['Close']

        c = c.squeeze()
        if isinstance(c, pd.DataFrame): 
            c = c.iloc[:, 0]
        
        returns = c.pct_change().dropna()
        
        # Volatility metrics
        daily_vol = returns.std() * 100
        annual_vol = daily_vol * np.sqrt(252)
        
        # Trend strength (ratio of up days)
        trend_strength = (returns > 0).sum() / len(returns)
        
        # Price range volatility
        price_range = (c.max() - c.min()) / c.mean()
        
        # Recent vs Historical Volatility
        recent_returns = returns.iloc[-60:] if len(returns) > 60 else returns
        recent_vol = recent_returns.std() * np.sqrt(252) * 100
        vol_drift = ((recent_vol - annual_vol) / annual_vol) * 100
        
        # 6-TIER REGIME CLASSIFICATION
        if annual_vol > 80: 
            regime = "extreme_volatility"    # Tier 1
        elif annual_vol > 50: 
            regime = "very_high_volatility"  # Tier 2
        elif annual_vol > 35: 
            regime = "high_volatility"       # Tier 3
        elif annual_vol > 20:
            regime = "medium_high_volatility"# Tier 4
        elif annual_vol > 12:
            regime = "medium_volatility"     # Tier 5
        else:
            regime = "low_volatility"        # Tier 6
        
        profile = {
            'ticker': ticker,
            'regime': regime,
            'daily_vol': daily_vol,
            'annual_vol': annual_vol,
            'recent_vol': recent_vol,
            'vol_drift': vol_drift,
            'trend_strength': trend_strength,
            'price_range': price_range
        }
        
        print(f"📊 {ticker} Profile: {regime.upper()} (Vol: {annual_vol:.1f}%)")
        return profile
    
    @staticmethod
    def get_hyperparameters(profile):
        """Return regime-specific hyperparameters (6-Tier System)"""
        regime = profile['regime']
        
        # TIER 1: EXTREME
        if regime == "extreme_volatility":
            return {
                'regime': regime,
                'hidden_gen': 32, 'hidden_spec': 64,
                'lookback': 30, 'epochs': 20, 'lr': 0.005,
                'changepoint_prior': 0.25, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.006, 'reward_scale': 70, 
                'ent_coef': 0.04, 'learning_rate': 0.0003, 'total_timesteps': 150000
            }
        
        # TIER 2: VERY HIGH
        elif regime == "very_high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 48, 'hidden_spec': 96,
                'lookback': 60,     
                'epochs': 25, 'lr': 0.003,
                'changepoint_prior': 0.08, 
                'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.006,  
                'reward_scale': 40,         
                'ent_coef': 0.005,          
                'learning_rate': 0.0002, 'total_timesteps': 100000
            }
        
        # TIER 3: HIGH
        elif regime == "high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 48, 'hidden_spec': 96,
                'lookback': 45, 'epochs': 25, 'lr': 0.003,
                'changepoint_prior': 0.10, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.0025, 'reward_scale': 120, 
                'ent_coef': 0.02, 'learning_rate': 0.00025, 'total_timesteps': 85000
            }
        
        # TIER 4: MEDIUM-HIGH
        elif regime == "medium_high_volatility":
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 50, 'epochs': 25, 'lr': 0.0025,
                'changepoint_prior': 0.05, 'seasonality_mode': 'multiplicative',
                'reward_threshold': 0.002, 'reward_scale': 140, 
                'ent_coef': 0.015, 'learning_rate': 0.0002, 'total_timesteps': 75000
            }
        
        # TIER 5: MEDIUM
        elif regime == "medium_volatility":
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 60, 'epochs': 30, 'lr': 0.002,
                'changepoint_prior': 0.05, 'seasonality_mode': 'additive',
                'reward_threshold': 0.0015, 'reward_scale': 160, 
                'ent_coef': 0.01, 'learning_rate': 0.0002, 'total_timesteps': 60000
            }
        
        # TIER 6: LOW
        else: 
            return {
                'regime': regime,
                'hidden_gen': 64, 'hidden_spec': 128,
                'lookback': 60, 'epochs': 30, 'lr': 0.002,
                'changepoint_prior': 0.03, 'seasonality_mode': 'additive',
                'reward_threshold': 0.001, 'reward_scale': 200, 
                'ent_coef': 0.01, 'learning_rate': 0.0002, 'total_timesteps': 50000
            }

# ==========================================
# 1. SENTIMENT ENGINE
# ==========================================
class SentimentEngine:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def scrape_news(self, ticker):
        search_ticker = ticker.replace(".NS", "")
        query = f"{search_ticker} stock news"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all('item')
            headlines = [item.title.text for item in items[:5]]
            return headlines
        except Exception:
            return []

    def get_sentiment(self, ticker):
        headlines = self.scrape_news(ticker)
        if not headlines:
            return 0.0, ["No Recent News Found"]
        results = self.nlp(headlines)
        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        total = sum([score_map[res['label']] * res['score'] for res in results])
        final = max(min(total / len(headlines), 1.0), -1.0)
        return final, headlines

# ==========================================
# 2. ADAPTIVE HYBRID FORECASTER (WITH ATTENTION)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attention(out)
        return self.fc(context)

class TridentForecaster:
    def __init__(self, ticker, general_ticker="SPY", hyperparams=None):
        self.ticker = ticker
        self.general_ticker = general_ticker
        self.hyperparams = hyperparams or {}
        
        # Adaptive parameters
        hidden_gen = self.hyperparams.get('hidden_gen', 32)
        hidden_spec = self.hyperparams.get('hidden_spec', 64)
        self.lookback = self.hyperparams.get('lookback', 60)
        
        # Neural Components
        self.general_model = AttentionLSTM(hidden_dim=hidden_gen).to(DEVICE)
        self.specialist_model = AttentionLSTM(hidden_dim=hidden_spec).to(DEVICE)
        self.scaler_gen = StandardScaler()
        self.scaler_spec = StandardScaler()
        
        # Structural Component
        self.prophet_model = None

    def get_data(self, start_date, end_date):
        gen_df = yf.download(self.general_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        spec_df = yf.download(self.ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        return gen_df, spec_df

    def _to_log_returns(self, prices):
        return np.diff(np.log(prices + 1e-9), axis=0)

    def train_prophet(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            try:
                close_data = df.xs(self.ticker, level=1, axis=1)['Close']
            except KeyError:
                 close_data = df['Close']
        else:
            close_data = df['Close']

        close_data = close_data.squeeze()
        if isinstance(close_data, pd.DataFrame): 
            close_data = close_data.iloc[:, 0]
        
        p_df = df.reset_index()
        date_col = 'Date' if 'Date' in p_df.columns else p_df.columns[0]
        
        p_df = pd.DataFrame({
            'ds': pd.to_datetime(p_df[date_col]),
            'y': close_data.values
        })
        
        # Adaptive Prophet configuration
        changepoint_prior = self.hyperparams.get('changepoint_prior', 0.05)
        seasonality_mode = self.hyperparams.get('seasonality_mode', 'multiplicative')
        
        self.prophet_model = Prophet(
            changepoint_prior_scale=changepoint_prior,
            seasonality_mode=seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.prophet_model.fit(p_df)

    def get_prophet_signal(self, df):
        if self.prophet_model is None: 
            return np.zeros(len(df))
        
        dates_df = df.reset_index()
        date_col = 'Date' if 'Date' in dates_df.columns else dates_df.columns[0]
        
        future = pd.DataFrame({'ds': pd.to_datetime(dates_df[date_col])})
        forecast = self.prophet_model.predict(future)
        
        # Trend returns
        trend_prices = forecast['yhat'].values
        trend_returns = np.diff(np.log(trend_prices + 1e-9))
        trend_returns = np.insert(trend_returns, 0, 0.0)
        
        # Robust standardization
        mean_ret = np.mean(trend_returns)
        std_ret = np.std(trend_returns)
        if std_ret < 1e-9:
            return np.zeros(len(trend_returns))
        
        return (trend_returns - mean_ret) / std_ret

    def train(self, gen_df, spec_df):
        if len(spec_df) < self.lookback + 10: 
            return False
        
        # 1. Train Prophet
        self.train_prophet(spec_df)
        
        # 2. Train LSTMs
        common_len = min(len(gen_df), len(spec_df))
        gen_raw = gen_df['Close'].values[-common_len:].reshape(-1, 1)
        spec_raw = spec_df['Close'].values[-common_len:].reshape(-1, 1)
        
        gen_ret = self._to_log_returns(gen_raw)
        spec_ret = self._to_log_returns(spec_raw)
        
        gen_norm = self.scaler_gen.fit_transform(gen_ret)
        spec_norm = self.scaler_spec.fit_transform(spec_ret)
        
        X_gen, y_gen = self._create_sequences(gen_norm)
        X_spec, y_spec = self._create_sequences(spec_norm)
        
        epochs = self.hyperparams.get('epochs', 20)
        lr = self.hyperparams.get('lr', 0.005)
        
        self._train_single(self.general_model, X_gen, y_gen, epochs=epochs, lr=lr)
        self._train_single(self.specialist_model, X_spec, y_spec, epochs=epochs, lr=lr)
        return True

    def predict(self, recent_gen, recent_spec):
        self.general_model.eval()
        self.specialist_model.eval()
        
        gen_ret = self._to_log_returns(recent_gen.reshape(-1, 1))
        spec_ret = self._to_log_returns(recent_spec.reshape(-1, 1))
        
        if len(gen_ret) < self.lookback: 
            return 0.0, 0.0
        
        # Input Normalization/Clipping
        gen_ret = np.clip(gen_ret, -5.0, 5.0)
        spec_ret = np.clip(spec_ret, -5.0, 5.0)
        
        gen_in = self.scaler_gen.transform(gen_ret[-self.lookback:].reshape(-1, 1)).reshape(1, self.lookback, 1)
        spec_in = self.scaler_spec.transform(spec_ret[-self.lookback:].reshape(-1, 1)).reshape(1, self.lookback, 1)
        
        with torch.no_grad():
            p_gen = self.general_model(torch.Tensor(gen_in).to(DEVICE)).cpu().item()
            p_spec = self.specialist_model(torch.Tensor(spec_in).to(DEVICE)).cpu().item()
        return p_gen, p_spec

    def batch_predict(self, gen_df, spec_df):
        self.general_model.eval()
        self.specialist_model.eval()
        
        common_len = min(len(gen_df), len(spec_df))
        gen_raw = gen_df['Close'].values[-common_len:].reshape(-1, 1)
        spec_raw = spec_df['Close'].values[-common_len:].reshape(-1, 1)
        
        gen_ret = self._to_log_returns(gen_raw)
        spec_ret = self._to_log_returns(spec_raw)
        
        # Input Normalization/Clipping
        gen_ret = np.clip(gen_ret, -5.0, 5.0)
        spec_ret = np.clip(spec_ret, -5.0, 5.0)
        
        gen_norm = self.scaler_gen.transform(gen_ret)
        spec_norm = self.scaler_spec.transform(spec_ret)
        
        X_gen, _ = self._create_sequences(gen_norm)
        X_spec, _ = self._create_sequences(spec_norm)
        
        with torch.no_grad():
            out_gen = self.general_model(torch.Tensor(X_gen).to(DEVICE)).cpu().numpy().flatten()
            out_spec = self.specialist_model(torch.Tensor(X_spec).to(DEVICE)).cpu().numpy().flatten()
        
        prophet_full = self.get_prophet_signal(spec_df.iloc[-common_len:])
        
        lstm_len = len(out_spec)
        if len(prophet_full) > lstm_len:
            prophet_aligned = prophet_full[-lstm_len:]
        elif len(prophet_full) < lstm_len:
            pad_len = lstm_len - len(prophet_full)
            prophet_aligned = np.concatenate([np.zeros(pad_len), prophet_full])
        else:
            prophet_aligned = prophet_full
        
        return out_gen, out_spec, prophet_aligned

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)

    def _train_single(self, model, X, y, epochs=20, lr=0.005):
        torch.manual_seed(SEED)
        X_t = torch.Tensor(X).to(DEVICE)
        y_t = torch.Tensor(y).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X_t)
            loss = crit(out, y_t)
            loss.backward()
            opt.step()

# ==========================================
# 3. ADAPTIVE RL ENVIRONMENT (ENHANCED)
# ==========================================
class TridentTradingEnv(gym.Env):
    def __init__(self, df, benchmark_df, signals, sentiment_score, hyperparams, training_mode=False):
        super(TridentTradingEnv, self).__init__()
        self.df = df
        self.benchmark_df = benchmark_df # Context
        self.gen_signals, self.spec_signals, self.prophet_signals = signals
        self.sentiment = sentiment_score
        self.hyperparams = hyperparams
        self.training_mode = training_mode
        self.np_random = np.random.RandomState(SEED)
        
        # CONTINUOUS ACTION SPACE: [-1, 1]
        # 1 = Full Long, 0.5 = 50% Long, 0 = Cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE: 12 Features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        c = df['Close'].squeeze()
        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
        
        # Base Indicators
        self.rsi = ta.momentum.rsi(c, window=14).fillna(50)
        self.sma50 = ta.trend.sma_indicator(c, window=50).fillna(c)
        
        # ENHANCED: Volume (OBV/VWAP proxies)
        v = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(np.zeros(len(c)))
        self.vol_ma5 = v.rolling(window=5).mean().fillna(0)
        self.vol_ma20 = v.rolling(window=20).mean().fillna(0)
        
        # ENHANCED: Volatility Term Structure
        r = c.pct_change().fillna(0)
        self.vol_short = r.rolling(window=5).std().fillna(0)
        self.vol_long = r.rolling(window=20).std().fillna(0)
        
        # ENHANCED: Benchmark (Market Breadth)
        b_c = benchmark_df['Close'].squeeze()
        if isinstance(b_c, pd.DataFrame): b_c = b_c.iloc[:,0]
        if len(b_c) > len(c): b_c = b_c.iloc[-len(c):]
        elif len(b_c) < len(c): 
            pad = pd.Series([0]*(len(c)-len(b_c)))
            b_c = pd.concat([pad, b_c], ignore_index=True)
        self.bench_ret = b_c.pct_change().fillna(0).reset_index(drop=True)
        
        self.balance = 10000
        self.shares = 0
        self.history = [10000]
        self.high_water_mark = 10000
        self.trades = []
        self.prev_action = 0.0 
        
        self.start_step = 62
        self.current_step = self.start_step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.balance = 10000
        self.shares = 0
        self.history = [10000]
        self.high_water_mark = 10000
        self.trades = []
        self.prev_action = 0.0
        
        if self.training_mode and len(self.df) > 500:
            self.current_step = self.np_random.randint(self.start_step, len(self.df) - 250)
        else:
            self.current_step = self.start_step
        
        return self._next_obs(), {}

    def _next_obs(self):
        price = float(self.df['Close'].values[self.current_step].item())
        
        offset = len(self.df) - len(self.gen_signals)
        sig_idx = self.current_step - offset
        
        if 0 <= sig_idx < len(self.gen_signals):
            g = self.gen_signals[sig_idx]
            s = self.spec_signals[sig_idx]
            p = self.prophet_signals[sig_idx]
        else:
            g, s, p = 0.0, 0.0, 0.0

        total = self.balance + (self.shares * price)
        alloc = (self.shares * price) / total if total > 0 else 0
        
        prev = float(self.df['Close'].values[self.current_step-1].item())
        ret = ((price - prev)/prev) * 100 
        
        recent = self.df['Close'].values[self.current_step-20:self.current_step]
        vol = (np.std(recent) / np.mean(recent)) * 100 if len(recent) > 0 else 0.0
        
        curr_rsi = self.rsi.values[self.current_step] / 100.0 
        curr_sma = self.sma50.values[self.current_step]
        trend_slope = ((price - curr_sma) / curr_sma) * 10 
        
        bench_r = self.bench_ret.iloc[self.current_step] * 100
        
        v5 = self.vol_ma5.iloc[self.current_step]
        v20 = self.vol_ma20.iloc[self.current_step]
        vol_osc = (v5 - v20) / (v20 + 1e-9)
        
        vr_short = self.vol_short.iloc[self.current_step]
        vr_long = self.vol_long.iloc[self.current_step]
        vol_ratio = vr_short / (vr_long + 1e-9)
        
        if self.training_mode:
            sent = self.sentiment if self.np_random.rand() > 0.3 else np.random.uniform(-1,1)
        else:
            sent = self.sentiment
        
        # Input Normalization
        obs = np.array([
            g, s, p, sent, alloc, 
            ret / 5.0,       # Scale returns down
            vol / 100.0,     # Scale vol down
            curr_rsi, 
            trend_slope / 10.0, 
            bench_r / 5.0,   # Scale bench ret
            vol_osc, 
            vol_ratio
        ], dtype=np.float32)
        
        # Clip for safety
        obs = np.clip(obs, -5.0, 5.0)
        return obs

    def step(self, action):
        target_alloc = np.clip(action[0], 0, 1) 
        price_t = float(self.df['Close'].values[self.current_step].item())
        
        total_val = self.balance + (self.shares * price_t)
        target_shares_val = total_val * target_alloc
        current_shares_val = self.shares * price_t
        diff = target_shares_val - current_shares_val
        fee = 0.001
        
        if diff > 0: 
            cost = diff * (1 + fee)
            if self.balance >= cost:
                shares_to_buy = int(diff / price_t)
                if shares_to_buy > 0:
                    self.balance -= (shares_to_buy * price_t) * (1+fee)
                    self.shares += shares_to_buy
                    if shares_to_buy > 0:
                        self.trades.append({'step': self.current_step, 'type': 'buy', 'price': price_t, 'amt': shares_to_buy})
        
        elif diff < 0: 
            shares_to_sell = int(abs(diff) / price_t)
            if shares_to_sell > 0:
                self.shares -= shares_to_sell
                self.balance += (shares_to_sell * price_t) * (1-fee)
                if shares_to_sell > 0:
                    self.trades.append({'step': self.current_step, 'type': 'sell', 'price': price_t, 'amt': shares_to_sell})

        self.current_step += 1
        
        truncated = False
        if self.current_step >= len(self.df) - 1:
            done = True
            price_next = price_t
        else:
            done = False
            price_next = float(self.df['Close'].values[self.current_step].item())
        
        if self.training_mode and (self.current_step - self.start_step > 300):
            truncated = True
        
        new_val = self.balance + (self.shares * price_next)
        self.history.append(new_val)
        
        # REWARD FUNCTION
        port_ret = np.log(new_val / total_val) if total_val > 0 else 0
        market_ret = (price_next - price_t) / price_t
        
        reward_scale = self.hyperparams.get('reward_scale', 100)
        threshold = self.hyperparams.get('reward_threshold', 0.002)
        
        # 1. Base Profit
        base_reward = port_ret * reward_scale
            
        # 2. Sizing Quality
        sizing_quality = target_alloc if market_ret > 0 else (1.0 - target_alloc)
        regime_bonus = 0.0
        if abs(market_ret) > threshold:
            regime_bonus = sizing_quality * 0.5
        
        # 3. Dynamic Churn Penalty
        current_regime = self.hyperparams.get('regime', 'low_volatility')
        base_penalty = -0.01
        if current_regime in ["extreme_volatility", "very_high_volatility"]:
             base_penalty = -0.025 
        
        delta_alloc = abs(target_alloc - self.prev_action)
        churn_penalty = base_penalty * delta_alloc 
        
        # 4. Holding Bonus
        holding_bonus = 0.0
        if target_alloc > 0.1 and market_ret > 0:
            holding_bonus = 0.5 * reward_scale * market_ret
            
        # 5. Wake Up Bonus
        wake_up_bonus = 0.0
        if target_alloc > 0.05:
            wake_up_bonus = 0.05 

        self.prev_action = target_alloc
        
        # Composite Reward
        reward = base_reward + regime_bonus + churn_penalty + holding_bonus + wake_up_bonus
        
        return self._next_obs(), reward, done or truncated, truncated, {}

# ==========================================
# 4. ADAPTIVE UTILITIES
# ==========================================
# MODIFIED: Single Model Source of Truth
def load_or_train_agent(ticker, df, benchmark_df, signals, hyperparams, retrain=False):
    clean_ticker = ticker.replace(".NS", "")
    # SINGLE FILENAME PER TICKER
    model_path = os.path.join(MODEL_DIR, f"agent_{clean_ticker}")
    
    def make_env():
        e = TridentTradingEnv(df, benchmark_df, signals, sentiment_score=0.0, hyperparams=hyperparams, training_mode=True)
        e.reset(seed=SEED)
        return e

    # 1. Try to Load (unless forced to retrain)
    if not retrain and os.path.exists(f"{model_path}.zip"):
        try:
            train_env = DummyVecEnv([make_env])
            model = PPO.load(model_path, env=train_env, device='cpu')
            # Ensure input dimensions match (avoids crash if features changed)
            if model.observation_space.shape[0] == 12:
                return model, "Loaded"
        except: 
            pass

    # 2. Train (if missing or forced)
    print(f"⚙️ Training Continuous Agent for {ticker}...")
    train_env = DummyVecEnv([make_env])
    
    ent_coef = hyperparams.get('ent_coef', 0.01)
    learning_rate = hyperparams.get('learning_rate', 0.0002)
    total_timesteps = hyperparams.get('total_timesteps', 50000)
    
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=0, 
        seed=SEED,
        device='cpu', 
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        batch_size=128,
        n_steps=2048
    )
    model.learn(total_timesteps=total_timesteps)
    
    # SAVE AS THE SINGLE SOURCE OF TRUTH
    model.save(model_path) 
    return model, "Trained"

def add_feature_engineering(df):
    c = df['Close'].squeeze()
    if isinstance(c, pd.DataFrame): 
        c = c.iloc[:,0]
    df['SMA_20'] = ta.trend.sma_indicator(c, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(c, window=50)
    df['RSI'] = ta.momentum.rsi(c, window=14)
    df['MACD'] = ta.trend.macd_diff(c)
    return df
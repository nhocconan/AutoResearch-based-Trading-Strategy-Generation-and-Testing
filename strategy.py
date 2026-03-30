#!/usr/bin/env python3
"""
Experiment #006: Funding Rate Mean-Reversion + 1w Trend Filter + Volume

HYPOTHESIS: Funding rate mean-reversion is the #1 proven edge for BTC/ETH.
Combined with weekly trend filter, this strategy should:
- Work through 2022 crash (funding extreme = reversals)
- Work through 2025 bear (funding < -0.1% = bounce setup)
- Generate 40-80 trades over 4 years (10-20/year)

WHY IT WORKS ON BOTH BULL AND BEAR:
- Bull: When funding drops below -0.2% (fear), price tends to recover
- Bear: When funding spikes above +0.2% (greed), shorts pay off
- 1w SMA200 filters out counter-trend trades (avoid long in bear)

FUNDING RATE DATA:
- Process with: pd.read_parquet('data/processed/funding/BTCUSDT_funding_1h.parquet')
- 1h funding rates aggregated to 1d for Z-score calculation
- Z-score of (funding - 30d SMA) / 30d STD

TARGET: 40-80 total trades over 4 years (10-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
import os

name = "funding_rate_mean_reversion_1w_trend"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(data, period=20):
    """Z-score of rolling window"""
    n = len(data)
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(data).rolling(window=period, min_periods=period).std().values
    std = np.where(std > 1e-10, std, 1e-10)
    return (data - sma) / std

def load_funding_data(symbol):
    """Load funding rate data for symbol"""
    funding_path = f'data/processed/funding/{symbol}_funding_1h.parquet'
    if not os.path.exists(funding_path):
        return None
    try:
        df = pd.read_parquet(funding_path)
        return df
    except:
        return None

def aggregate_funding_to_daily(funding_df, prices):
    """Aggregate 1h funding rates to daily"""
    if funding_df is None:
        return np.full(len(prices), 0.0)
    
    # Parse timestamp
    funding_df = funding_df.copy()
    if 'open_time' in funding_df.columns:
        funding_df['timestamp'] = pd.to_datetime(funding_df['open_time'])
    elif 'timestamp' in funding_df.columns:
        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'])
    else:
        return np.full(len(prices), 0.0)
    
    # Resample to daily
    funding_df = funding_df.set_index('timestamp')
    
    # Get funding rate column
    if 'funding_rate' in funding_df.columns:
        rate_col = 'funding_rate'
    elif 'rate' in funding_df.columns:
        rate_col = 'rate'
    else:
        return np.full(len(prices), 0.0)
    
    # Daily mean funding rate
    daily_funding = funding_df[rate_col].resample('1D').mean()
    
    # Align to price timestamps
    prices_idx = pd.DatetimeIndex(prices['open_time'])
    daily_funding_aligned = daily_funding.reindex(prices_idx, method='ffill').fillna(0.0)
    
    return daily_funding_aligned.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA(200) for trend - simple version is more robust
    sma_200_1w = calculate_sma(df_1w['close'].values, period=200)
    sma_200_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # === Load funding rate data ===
    symbol = prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'
    funding_df = load_funding_data(symbol)
    daily_funding = aggregate_funding_to_daily(funding_df, prices)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Funding rate Z-score (30d)
    funding_zscore = calculate_zscore(daily_funding, period=30)
    
    # Volume ratio (20d MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 350  # 200 for 1w SMA + 30 for Z-score + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND FILTER ===
        # Only trade in direction of weekly trend
        weekly_trend_up = close[i] > sma_200_aligned[i]
        weekly_trend_down = close[i] < sma_200_aligned[i]
        
        # === FUNDING RATE Z-SCORE (primary signal) ===
        fz = funding_zscore[i]
        
        # Long when funding severely depressed (fear = bounce)
        long_signal = fz < -1.8 and weekly_trend_up
        
        # Short when funding severely elevated (greed = reversal)
        short_signal = fz > 1.8 and weekly_trend_down
        
        # === VOLUME CONFIRMATION (optional but helps) ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Long: funding Z < -1.8 + weekly trend up
            if long_signal and vol_confirm:
                desired_signal = SIZE
            
            # Short: funding Z > +1.8 + weekly trend down
            if short_signal and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5x ATR) ===
        if in_position:
            if position_side > 0:
                # Long stoploss
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_down:
                    desired_signal = 0.0
                
                # Exit if funding mean-reverts (Z > 0.5)
                if funding_zscore[i] > 0.5:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stoploss
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_up:
                    desired_signal = 0.0
                
                # Exit if funding mean-reverts (Z < -0.5)
                if funding_zscore[i] < -0.5:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 days to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
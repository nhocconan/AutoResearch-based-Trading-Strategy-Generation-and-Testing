#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week volume confirmation and 1-month EMA trend filter
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
# Uses weekly volume spike and monthly EMA to avoid false breakouts and catch major trends

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    
    # Load 1m data for trend filter (EMA20)
    df_1m = get_htf_data(prices, '1m')
    if len(df_1m) < 20:
        return np.zeros(n)
    close_1m = df_1m['close'].values
    
    # Volume average (10-period on 1w)
    vol_avg_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # EMA20 on 1m for trend filter
    ema20_1m = pd.Series(close_1m).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR for volatility (14-period on 1d)
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    ema20_1m_aligned = align_htf_to_ltf(prices, df_1m, ema20_1m)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(150, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_avg_1w_aligned[i]) or np.isnan(ema20_1m_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (close[i] > donch_high_aligned[i] and 
            close[i] > ema20_1m_aligned[i] and 
            volume[i] > 2.0 * vol_avg_1w_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (close[i] < donch_low_aligned[i] and 
              close[i] < ema20_1m_aligned[i] and 
              volume[i] > 2.0 * vol_avg_1w_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < ema20_1m_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema20_1m_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_1wVolume_1mEMA_Breakout"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d EMA200 trend filter
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Uses volume spike and long-term trend filter to avoid false breakouts
# Targets BTC/ETH primarily - avoids overtrading with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 6h Donchian channels (20-period)
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Volume average (20-period on 12h)
    vol_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # EMA200 on 1d for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # ATR for volatility and position sizing (14-period on 6h)
    tr1 = np.maximum(high_6h[1:], low_6h[:-1]) - np.minimum(high_6h[1:], low_6h[:-1])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.30)
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (close[i] > donch_high_aligned[i] and 
            close[i] > ema200_1d_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (close[i] < donch_low_aligned[i] and 
              close[i] < ema200_1d_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or trend reversal
        elif position == 1 and close[i] < ema200_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema200_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_12hVolume_1dEMA200_Trend"
timeframe = "6h"
leverage = 1.0
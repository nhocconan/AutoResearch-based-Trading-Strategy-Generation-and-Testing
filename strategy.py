#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w volume confirmation and 1d trend filter
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Uses volume spike and EMA trend filter to avoid false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Volume average (20-period on 1w)
    vol_avg = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and stoploss (14-period on 12h)
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(high_12h[1:], low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (close[i] > donch_high_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (close[i] < donch_low_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or volatility-based stop
        elif position == 1 and (close[i] < ema50_1d_aligned[i] or 
                                close[i] < donch_low_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema50_1d_aligned[i] or 
                                 close[i] > donch_high_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1wVolume_1dEMA_Breakout"
timeframe = "12h"
leverage = 1.0
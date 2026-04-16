#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian upper and lower bands (20 periods)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # 12h EMA20 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 1w data (HTF for regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w ATR for volatility filter
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 1d data (HTF for volume filter) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = volume_1d / (vol_ma_10_1d + 1e-10)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR volatility too high
            if (close[i] < donchian_lower_12h[i]) or (atr_1w_aligned[i] > np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR volatility too high
            if (close[i] > donchian_upper_12h[i]) or (atr_1w_aligned[i] > np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above EMA20 (trend filter) 
            # AND volume spike AND volatility not too high
            if (close[i] > donchian_upper_12h[i]) and (close[i] > ema_20_12h_aligned[i]) and \
               (vol_ratio_1d_aligned[i] > 1.5) and (atr_1w_aligned[i] < np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below EMA20 (trend filter) 
            # AND volume spike AND volatility not too high
            elif (close[i] < donchian_lower_12h[i]) and (close[i] < ema_20_12h_aligned[i]) and \
                 (vol_ratio_1d_aligned[i] > 1.5) and (atr_1w_aligned[i] < np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA20_Volume"
timeframe = "12h"
leverage = 1.0
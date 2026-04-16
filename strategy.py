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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian upper and lower bands (15 periods)
    high_15_12h = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    low_15_12h = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_15_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_15_12h)
    
    # 12h EMA25 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_25_12h = close_12h_series.ewm(span=25, min_periods=25, adjust=False).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # === 1w data (HTF for regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w ATR for volatility regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Volume spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_25_12h_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_25_12h_val = ema_25_12h_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR volatility regime shifts to high
            if (price < lower_12h) or (atr_1w_val > np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR volatility regime shifts to high
            if (price > upper_12h) or (atr_1w_val > np.percentile(atr_1w_aligned[:i+1], 80)):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above EMA25 (trend filter) 
            # AND volatility regime is low AND volume spike
            if (price > upper_12h) and (price > ema_25_12h_val) and \
               (atr_1w_val < np.percentile(atr_1w_aligned[:i+1], 50)) and \
               (vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below EMA25 (trend filter) 
            # AND volatility regime is low AND volume spike
            elif (price < lower_12h) and (price < ema_25_12h_val) and \
                 (atr_1w_val < np.percentile(atr_1w_aligned[:i+1], 50)) and \
                 (vol_ratio_val > 1.8):
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

name = "12h_Donchian_Breakout_EMA25_VolRegime"
timeframe = "12h"
leverage = 1.0
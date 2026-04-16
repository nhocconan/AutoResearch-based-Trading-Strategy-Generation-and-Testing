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
    
    # 12h Donchian(15) for breakout levels
    high_15_12h = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    low_15_12h = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_15_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_15_12h)
    
    # 12h ATR for volatility filter
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1w data (HTF for regime filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w ATR for volatility regime
    tr1w = np.abs(high_1w - low_1w)
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2w[0] = np.inf
    tr3w[0] = np.inf
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = pd.Series(trw).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 1d volume for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR volatility regime shifts
            if (price < lower_12h) or (atr_12h_val > 1.5 * atr_1w_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR volatility regime shifts
            if (price > upper_12h) or (atr_12h_val > 1.5 * atr_1w_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above daily EMA34 (trend filter) 
            # AND volatility contraction (low 12h ATR relative to 1w ATR) AND volume surge
            if (price > upper_12h) and (price > ema_34_1d_val) and \
               (atr_12h_val < 0.7 * atr_1w_val) and (vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below daily EMA34 (trend filter) 
            # AND volatility contraction AND volume surge
            elif (price < lower_12h) and (price < ema_34_1d_val) and \
                 (atr_12h_val < 0.7 * atr_1w_val) and (vol_ratio_val > 2.5):
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

name = "12h_Donchian_Breakout_EMA34_1d_Vol_VolatilityFilter"
timeframe = "12h"
leverage = 1.0
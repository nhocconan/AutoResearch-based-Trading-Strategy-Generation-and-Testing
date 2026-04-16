#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (for daily levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian(20) channels
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = align_htf_to_ltf(prices, df_1d, high_20_1d)
    donchian_lower_1d = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w data (for trend confirmation) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike detection (daily)
    vol_ma_5_1d = pd.Series(volume_1d).rolling(window=5, min_periods=5).mean().values
    vol_ratio_1d = volume_1d / vol_ma_5_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_1d = donchian_upper_1d[i]
        lower_1d = donchian_lower_1d[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_200_1w_val = ema_200_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR below daily EMA50
            if (price < lower_1d) or (price < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR above daily EMA50
            if (price > upper_1d) or (price > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above daily EMA50 AND above weekly EMA200
            # AND volume spike AND volatility not extreme
            if (price > upper_1d) and (price > ema_50_1d_val) and (price > ema_200_1w_val) and \
               (vol_ratio_val > 1.5) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 70)):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian lower AND below daily EMA50 AND below weekly EMA200
            # AND volume spike AND volatility not extreme
            elif (price < lower_1d) and (price < ema_50_1d_val) and (price < ema_200_1w_val) and \
                 (vol_ratio_val > 1.5) and (atr_1d_val < np.percentile(atr_1d_aligned[:i+1], 70)):
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

name = "12h_Donchian_Breakout_EMA50_1d_EMA200_1w_Volume_Filter"
timeframe = "12h"
leverage = 1.0
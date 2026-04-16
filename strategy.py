#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly OHLC for Pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point and key levels (R2/S2)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r2_1w = pivot_1w + range_1w * 1.1 / 6  # Weekly R2
    s2_1w = pivot_1w - range_1w * 1.1 / 6  # Weekly S2
    
    # === Daily ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly levels to 6h timeframe
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Price distance from weekly pivot (avoid chop) ===
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    dist_from_pivot = np.abs(close - pivot_6h)
    avg_dist = pd.Series(dist_from_pivot).rolling(window=50, min_periods=50).mean().values
    too_close = dist_from_pivot < (0.3 * avg_dist)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(atr_6h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(too_close[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r2_level = r2_6h[i]
        s2_level = s2_6h[i]
        atr_avg = atr_6h[i]
        vol_spike = volume_spike[i]
        too_close_to_pivot = too_close[i]
        
        # === EXIT LOGIC: Exit when price moves against position or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below S2 or volatility drops significantly
            if price < s2_level or atr_avg < (atr_6h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R2 or volatility drops significantly
            if price > r2_level or atr_avg < (atr_6h[i-1] * 0.7 if i > 0 else atr_avg):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly R2 with volume spike, sufficient volatility, and not too close to pivot
            if price > r2_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly S2 with volume spike, sufficient volatility, and not too close to pivot
            elif price < s2_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
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

name = "6h_Weekly_R2_S2_Breakout_Volume_ATRFilter_DistFilter"
timeframe = "6h"
leverage = 1.0
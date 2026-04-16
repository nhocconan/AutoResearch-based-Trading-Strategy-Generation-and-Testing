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
    
    # === Weekly OHLC for pivot calculation (HTF = 1w) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (Standard)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_hl_1w = high_1w - low_1w
    r2_1w = pivot_1w + range_hl_1w
    s2_1w = pivot_1w - range_hl_1w
    r3_1w = pivot_1w + 2 * range_hl_1w
    s3_1w = pivot_1w - 2 * range_hl_1w
    
    # === ATR for volatility filter (14-period on 6h) ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_6h = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    
    # Align Weekly Pivot levels to 6h timeframe
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    atr_ma_6h_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_6h)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(atr_ma_6h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r2_level = r2_6h[i]
        s2_level = s2_6h[i]
        r3_level = r3_6h[i]
        s3_level = s3_6h[i]
        atr_ma = atr_ma_6h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when price moves against position beyond S3/R3 or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below S3 or volatility drops significantly
            if price < s3_level or atr_ma < (atr_ma_6h_aligned[i-1] * 0.6 if i > 0 else atr_ma):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or volatility drops significantly
            if price > r3_level or atr_ma < (atr_ma_6h_aligned[i-1] * 0.6 if i > 0 else atr_ma):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R2 with volume spike, sufficient volatility
            if price > r2_level and vol_spike and atr_ma > 0:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S2 with volume spike, sufficient volatility
            elif price < s2_level and vol_spike and atr_ma > 0:
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

name = "6h_Weekly_Pivot_R2_S2_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0
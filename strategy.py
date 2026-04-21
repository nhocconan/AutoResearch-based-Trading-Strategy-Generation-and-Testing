#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_VolumeATRFilter_1wHTF
Hypothesis: 1d Camarilla pivot breakouts at R1/S1 with volume confirmation (>1.3x 20-period volume MA) and 1-week EMA50 trend filter. 
Uses 1d primary timeframe with 1w HTF for EMA50 trend alignment. Targets 7-25 trades/year (30-100 total over 4 years).
Volume confirmation ensures institutional participation, 1w EMA50 filter avoids counter-trend trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Indicators (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) 
            or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + uptrend (price > 1w EMA50)
            if price > camarilla_r1[i] and vol_ok and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + downtrend (price < 1w EMA50)
            elif price < camarilla_s1[i] and vol_ok and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1w EMA50
            if price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1w EMA50
            if price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_VolumeATRFilter_1wHTF"
timeframe = "1d"
leverage = 1.0
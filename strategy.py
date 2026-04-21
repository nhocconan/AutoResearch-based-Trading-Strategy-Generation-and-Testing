#!/usr/bin/env python3
"""
1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1
Hypothesis: 1d Camarilla pivot breakouts at R1/S1 with volume spike (>1.5x 20-period volume MA) and 1w HTF trend filter (price > weekly EMA21 for longs, < for shorts). 
Uses 1w HTF for EMA21 trend filter to ensure alignment with higher timeframe direction. 
Tight entry conditions to avoid overtrading. Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL. 
Works in bull/bear via 1w trend filter - only takes longs in weekly uptrend, shorts in weekly downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots, 1w for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 1d timeframe (no alignment needed - same timeframe)
    camarilla_r1_aligned = camarilla_r1  # Already 1d
    camarilla_s1_aligned = camarilla_s1  # Already 1d
    
    # === 1w EMA21 for trend filter ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === 1d Indicators (primary timeframe) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup period for volume MA
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        weekly_uptrend = price > ema_21_1w_aligned[i]
        weekly_downtrend = price < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + weekly uptrend
            if price > camarilla_r1_aligned[i] and vol_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + weekly downtrend
            elif price < camarilla_s1_aligned[i] and vol_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or weekly trend turns down
            if price < camarilla_s1_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or weekly trend turns up
            if price > camarilla_r1_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_VolumeSpike_EMAFilter_V1
Hypothesis: 1h Camarilla pivot breakouts at R1/S1 with volume spike confirmation (>1.5x 20-period volume MA) and EMA50 trend filter (price > EMA50 for longs, < for shorts). 
Camarilla levels from 4h HTF provide institutional support/resistance. Volume spikes confirm institutional participation. 
EMA50 trend filter ensures alignment with short-medium term trend. Session filter (08-20 UTC) reduces noise trades. 
Target 15-37 trades/year (60-150 total over 4 years) for 1h timeframe.
Uses 1h primary timeframe with 4h HTF for Camarilla calculation and EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for Camarilla pivots and EMA trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla levels
    camarilla_r1 = close_4h + (range_4h * 1.1 / 12.0)
    camarilla_s1 = close_4h - (range_4h * 1.1 / 12.0)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # === 4h EMA50 for trend filter ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Indicators (primary timeframe) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + uptrend (price > EMA50)
            if price > camarilla_r1_aligned[i] and vol_ok and price > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike + downtrend (price < EMA50)
            elif price < camarilla_s1_aligned[i] and vol_ok and price < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below EMA50 or volume spike fails
            if price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above EMA50 or volume spike fails
            if price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_VolumeSpike_EMAFilter_V1"
timeframe = "1h"
leverage = 1.0
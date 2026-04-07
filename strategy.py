#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance for 12h price action.
Long when price touches S3 level with volume confirmation and above 1d EMA50.
Short when price touches R3 level with volume confirmation and below 1d EMA50.
Works in both bull and bear markets by fading extremes at proven institutional levels.
Designed for 15-30 trades/year on 12h timeframe with clear pivot-based logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla formulas
    # R4 = C + ((H-L) * 1.5)
    # R3 = C + ((H-L) * 1.25)
    # R2 = C + ((H-L) * 1.166)
    # R1 = C + ((H-L) * 1.083)
    # S1 = C - ((H-L) * 1.083)
    # S2 = C - ((H-L) * 1.166)
    # S3 = C - ((H-L) * 1.25)
    # S4 = C - ((H-L) * 1.5)
    
    rng = ph - pl
    r3 = pc + (rng * 1.25)
    s3 = pc - (rng * 1.25)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align all 1d data to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price proximity to pivot levels (within 0.5%)
        proximity_threshold = 0.005
        near_s3 = abs(close[i] - s3_12h[i]) / s3_12h[i] < proximity_threshold
        near_r3 = abs(close[i] - r3_12h[i]) / r3_12h[i] < proximity_threshold
        
        # 1d trend filter
        above_ema50 = close[i] > ema50_12h[i]
        below_ema50 = close[i] < ema50_12h[i]
        
        if position == 1:  # Long position
            # Exit: price moves below S3 or trend turns bearish
            if close[i] < s3_12h[i] or below_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R3 or trend turns bullish
            if close[i] > r3_12h[i] or above_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near S3 with volume confirmation and bullish trend
            if near_s3 and vol_confirmed and above_ema50:
                position = 1
                signals[i] = 0.25
            # Short: price near R3 with volume confirmation and bearish trend
            elif near_r3 and vol_confirmed and below_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals
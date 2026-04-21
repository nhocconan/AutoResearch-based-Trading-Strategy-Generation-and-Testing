#!/usr/bin/env python3
"""
4h_HTF_1w_Camarilla_R3S3_Breakout_Volume_ATRStop_V1
Hypothesis: Use weekly Camarilla R3/S3 levels + 4h volume spike (>2x 20-bar MA) for breakout entry + ATR(14) stoploss (2.0x). 
Weekly levels provide stronger support/resistance than daily, reducing false breakouts in choppy markets. 
Volume spike confirms institutional participation. ATR stop manages risk in volatile conditions. 
Discrete position sizing (0.30) minimizes fee churn. Target 15-25 trades/year per symbol. 
Works in bull (breakouts hold) and bear (tight stops limit losses) via volume confirmation and weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly Camarilla pivots
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Levels (R3, S3) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla levels
    camarilla_r3 = pivot + 1.1 * (high_1w - low_1w) / 2.0
    camarilla_s3 = pivot - 1.1 * (high_1w - low_1w) / 2.0
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above weekly Camarilla R3 with volume spike
            if price > camarilla_r3_aligned[i-1] and vol_ok:
                signals[i] = 0.30
                position = 1
            # Short: break below weekly Camarilla S3 with volume spike
            elif price < camarilla_s3_aligned[i-1] and vol_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or price re-enters weekly range
            if price < camarilla_r3_aligned[i-1] - 2.0 * atr[i] or price < camarilla_s3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: ATR stoploss or price re-enters weekly range
            if price > camarilla_s3_aligned[i-1] + 2.0 * atr[i] or price > camarilla_r3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_HTF_1w_Camarilla_R3S3_Breakout_Volume_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
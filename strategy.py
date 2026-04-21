#!/usr/bin/env python3
"""
4h_HTF_Camarilla_Pivot_Volume_Spike_ATRStop_V1
Hypothesis: Use 1d Camarilla R3/S3 levels from daily pivot + 4h volume spike (>2x 20-bar MA) for breakout confirmation + ATR(14) stoploss (2.0x). Camarilla levels provide high-probability intraday support/resistance, volume spike filters weak breakouts, ATR stop manages risk. Designed for both bull (catch momentum) and bear (fade false breaks via tight stops) markets. Target 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for daily Camarilla pivots
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    camarilla_r3 = pivot + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = pivot - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
            # Long: break above Camarilla R3 with volume spike
            if price > camarilla_r3_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 with volume spike
            elif price < camarilla_s3_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r3_aligned[i-1] - 2.0 * atr[i] or (price < camarilla_s3_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s3_aligned[i-1] + 2.0 * atr[i] or (price > camarilla_r3_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Camarilla_Pivot_Volume_Spike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
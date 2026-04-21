#!/usr/bin/env python3
"""
12h_HTF_1w_Camarilla_Pivot_VolumeSpike_ATRStop_V1
Hypothesis: Use 1w Camarilla R4/S4 levels from weekly pivot + 12h volume spike (>2x 20-bar MA) for breakout confirmation + ATR(14) stoploss (2.0x). Weekly Camarilla levels provide high-probability support/resistance for swing trades, volume spike filters weak breakouts, ATR stop manages risk. Designed for both bull (catch momentum) and bear (fade false breaks via tight stops) markets. Target 12-37 trades/year per symbol.
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
    
    # === 1w Camarilla Pivot Levels (R4, S4) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla levels
    camarilla_r4 = pivot + 1.1 * (high_1w - low_1w) / 2.0
    camarilla_s4 = pivot - 1.1 * (high_1w - low_1w) / 2.0
    
    # Align to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === 12h Indicators ===
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
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Camarilla R4 with volume spike
            if price > camarilla_r4_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S4 with volume spike
            elif price < camarilla_s4_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r4_aligned[i-1] - 2.0 * atr[i] or (price < camarilla_s4_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s4_aligned[i-1] + 2.0 * atr[i] or (price > camarilla_r4_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1w_Camarilla_Pivot_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0
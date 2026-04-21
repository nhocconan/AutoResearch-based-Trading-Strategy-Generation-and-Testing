#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeATRFilter_Tight
Hypothesis: 12h breakout above 1d Camarilla R1 or below S1 with volume confirmation (>1.5x 20-bar MA) and ATR stoploss (2.0x) works on 12h timeframe. Uses 1d for pivot calculation. Tight entries target 12-37 trades/year per symbol. Works in bull/bear via mean-reversion exits at opposite Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla pivots
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1/S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 2  # R1 = close + 1.1*range/2
    camarilla_s1 = prev_close - range_1d * 1.1 / 2  # S1 = close - 1.1*range/2
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume MA (20-period)
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
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: break above R1 with volume
            if price > camarilla_r1_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < camarilla_s1_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: mean reversion to S1 or ATR stoploss
            if price < camarilla_s1_aligned[i] or price < close[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: mean reversion to R1 or ATR stoploss
            if price > camarilla_r1_aligned[i] or price > close[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeATRFilter_Tight"
timeframe = "12h"
leverage = 1.0
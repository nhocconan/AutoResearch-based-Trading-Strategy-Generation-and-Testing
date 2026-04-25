#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSp
Hypothesis: 4h Camarilla R1/S1 breakout with 1d ATR trend filter and volume confirmation.
Long when price breaks above R1 with 1d ATR expansion (volatility increase) and volume spike.
Short when price breaks below S1 with 1d ATR expansion and volume spike.
ATR filter ensures we trade during volatile regimes, avoiding low-volatility chop.
Volume confirms institutional participation. Works in both bull and bear markets by
focusing on volatility expansion breakouts rather than direction.
Target: 20-50 trades/year (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1. Calculate Camarilla pivot levels from previous 1d bar
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 2. 1d ATR(14) for volatility regime filter
    # ATR = EMA of True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR expansion: current ATR > 1.2x ATR 10 periods ago
    atr_ma_10 = pd.Series(atr_14_1d).rolling(window=10, min_periods=10).mean().values
    atr_expansion = atr_14_1d > (1.2 * atr_ma_10)
    
    # Align ATR expansion to 4h timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # 3. Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) + MA(10) + volume MA(20)
    start_idx = max(14 + 10, 20) + 1  # +1 for rolling mean
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(atr_expansion_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + ATR expansion + volume spike
            long_setup = (close[i] > r1_1d_aligned[i]) and atr_expansion_aligned[i] and volume_spike[i]
            # Short: price breaks below S1 + ATR expansion + volume spike
            short_setup = (close[i] < s1_1d_aligned[i]) and atr_expansion_aligned[i] and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR ATR contraction
            if (close[i] < s1_1d_aligned[i]) or (not atr_expansion_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR ATR contraction
            if (close[i] > r1_1d_aligned[i]) or (not atr_expansion_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0
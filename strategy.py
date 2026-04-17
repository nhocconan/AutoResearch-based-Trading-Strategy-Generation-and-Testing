#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter
Strategy: Use 1d Camarilla pivot levels (R1/S1) for breakout entries with volume and ATR confirmation.
- Long when price breaks above R1 with volume > 1.5x 20-period average and ATR(14) > 0.5 * ATR(50)
- Short when price breaks below S1 with same filters
- Exit when price returns to the pivot point (PP) or opposite Camarilla level (S1 for long, R1 for short)
- Position size: 0.25
- Uses 1d Camarilla levels for structure, 6s for entry timing, volume/ATR for confirmation
- Works in bull/bear: breaks of key levels with volume/volatility confirmation capture strong moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels: R1, S1, PP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = PP + (Range * 1.1 / 2)
    r1_1d = pp_1d + (range_1d * 1.1 / 2)
    # S1 = PP - (Range * 1.1 / 2)
    s1_1d = pp_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6s
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # ATR for volatility filter
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for ATR50
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma20_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20_aligned[i])
        
        # ATR filter: short-term ATR > 50% of long-term ATR (volatility expansion)
        atr_filter = atr14[i] > (0.5 * atr50[i])
        
        # Entry conditions
        long_entry = (close[i] > r1_aligned[i]) and volume_filter and atr_filter
        short_entry = (close[i] < s1_aligned[i]) and volume_filter and atr_filter
        
        # Exit conditions
        long_exit = (position == 1) and (close[i] <= pp_aligned[i])
        short_exit = (position == -1) and (close[i] >= pp_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0
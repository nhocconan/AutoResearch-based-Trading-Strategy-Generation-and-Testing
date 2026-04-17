#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Camarilla pivot R1/S1 breakout + 1d volume spike + 1d EMA34 trend filter.
Long when price breaks above R1 with volume > 2.0x 20-period volume average and close > EMA34.
Short when price breaks below S1 with volume > 2.0x 20-period volume average and close < EMA34.
Uses 1d timeframe for pivot/volume/EMA to reduce noise and avoid overtrading on 6h.
Designed to capture intraday momentum with higher timeframe confirmation for better win rate.
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
    
    # Get 1d data for Camarilla pivots, volume, and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + 1.1 * (H - L) / 12
    # S1 = C - 1.1 * (H - L) / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-period average (scaled to 6h)
        # Scale 1d volume average to 6h: 1d has 4x 6h bars, so divide by 4
        vol_ma_20_6h_scaled = vol_ma_20_1d_aligned[i] / 4.0
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_6h_scaled
        
        if position == 0:
            # Long: price breaks above R1 with volume and close > EMA34
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and close < EMA34
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (opposite side)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 (opposite side)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarilla_R1S1_VolumeSpike_EMA34Filter"
timeframe = "6h"
leverage = 1.0
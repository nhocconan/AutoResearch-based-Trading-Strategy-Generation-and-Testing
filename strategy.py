#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with Volume Spike and 1D Trend Filter
Long: Price breaks above 1D Camarilla R1 + volume > 2x 4h volume MA + price > 1D EMA50
Short: Price breaks below 1D Camarilla S1 + volume > 2x 4h volume MA + price < 1D EMA50
Exit: Opposite break of Camarilla S1 (long) or R1 (short)
Uses Camarilla pivot levels from 1d for structure, volume spike for confirmation, 1D EMA50 for trend filter
Target: 25-35 trades/year per symbol
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
    
    # Get 1D data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla pivot levels for each 1D bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1D indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h volume moving average (20-period for confirmation)
    df_4h = get_htf_data(prices, '4h')
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_4h = align_htf_to_ltf(prices, df_4h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_4h[i]
        
        if position == 0:
            # Long: break above 1D Camarilla R1 + volume spike + 1D uptrend
            if price > r1_aligned[i] and vol > 2.0 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below 1D Camarilla S1 + volume spike + 1D downtrend
            elif price < s1_aligned[i] and vol > 2.0 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below 1D Camarilla S1 (reversal signal)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 1D Camarilla R1 (reversal signal)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_1DTrend"
timeframe = "4h"
leverage = 1.0
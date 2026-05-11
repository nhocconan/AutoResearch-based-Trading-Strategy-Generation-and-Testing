#!/usr/bin/env python3
"""
12h_12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v1
Hypothesis: Use Camarilla pivot levels from daily timeframe with R1/S1 levels as support/resistance.
Combine with 1d EMA34 trend filter and volume confirmation for breakout entries.
In trending markets, price breaks above R1 in uptrend or below S1 in downtrend with volume confirmation.
Camarilla levels provide precise turning points, and daily EMA34 filters for trend direction.
Target: 50-150 total trades over 4 years on 12h timeframe with tight entry conditions.
"""

name = "12h_12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1D Data for Camarilla and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = close_1d + range_hl * 1.1 / 12
    s1 = close_1d - range_hl * 1.1 / 12
    
    # Calculate EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume SMA20 for volume confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_sma20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA34
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_sma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > vol_sma20_aligned[i] * 1.5
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema34_aligned[i]
        price_below_ema = close[i] < ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in uptrend
            if close[i] > r1_aligned[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation in downtrend
            elif close[i] < s1_aligned[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or loses volume confirmation
            if close[i] < s1_aligned[i] or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 or loses volume confirmation
            if close[i] > r1_aligned[i] or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
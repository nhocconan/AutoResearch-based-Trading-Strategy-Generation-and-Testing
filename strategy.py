#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_With_1dTrendAndVolume
Hypothesis: 6h price breaks above/below Camarilla R1/S1 levels (from 1d pivot) with volume spike and 1d EMA34 trend confirmation.
Captures breakouts in trending markets while avoiding false moves. Works in bull (upside breakouts) and bear (downside breakouts).
Designed for 15-30 trades/year to minimize fee drag while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below S1 OR trend turns down
            if price < s1:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above R1 OR trend turns up
            if price > r1:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_With_1dTrendAndVolume"
timeframe = "6h"
leverage = 1.0
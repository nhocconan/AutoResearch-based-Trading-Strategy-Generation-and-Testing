#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_R2S2_Breakout_Volume_Trend_Filter
Hypothesis: Price breaks above Camarilla R1/S1 or R2/S2 levels with volume confirmation and EMA trend filter.
Camarilla levels derived from prior day's range provide institutional support/resistance.
Works in bull markets by buying breakouts above resistance, in bear markets by selling breakdowns below support.
Volume confirms institutional participation. EMA20 ensures trend alignment to avoid counter-trend trades.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    R2 = close_1d + 1.1 * (high_1d - low_1d) / 6
    S2 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align to 4h timeframe (use prior day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        ema20 = ema_20[i]
        
        if position == 0:
            # Long: price breaks above R1 or R2 with volume in uptrend
            if ((price > R1_aligned[i] or price > R2_aligned[i]) and vol_ok and price > ema20):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 or S2 with volume in downtrend
            elif ((price < S1_aligned[i] or price < S2_aligned[i]) and vol_ok and price < ema20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or trend reverses
            if price < S1_aligned[i] or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or trend reverses
            if price > R1_aligned[i] or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_R2S2_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0
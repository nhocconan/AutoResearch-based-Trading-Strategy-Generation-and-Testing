#!/usr/bin/env python3
"""
12h_VolumeSpike_Breakout_Direction
Hypothesis: Price breaking above/below 12h high/low with volume spike and confirmed direction from 1d EMA34 trend works in bull/bear markets.
Breakouts capture momentum; volume confirmation avoids false signals; 1d trend filter ensures alignment with higher timeframe trend.
Position size: ±0.25. Designed for fewer trades (<50/year) to minimize fee drag.
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
    
    # Calculate 12-period high/low for breakout detection
    high_max = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_min = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume spike detection: current volume > 2.0 x 24-period average
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(12, 24, 34)  # breakout lookback, volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(volume_ma24[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0 x 24-period average
        volume_filter = volume[i] > (2.0 * volume_ma24[i])
        
        # Breakout conditions
        breakout_up = high[i] > high_max[i-1]  # using previous bar's max to avoid look-ahead
        breakout_down = low[i] < low_min[i-1]  # using previous bar's min to avoid look-ahead
        
        if position == 0:
            # Long: upward breakout + volume filter + 1d uptrend
            if breakout_up and volume_filter and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume filter + 1d downtrend
            elif breakout_down and volume_filter and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downward breakout
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: upward breakout
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolumeSpike_Breakout_Direction"
timeframe = "12h"
leverage = 1.0
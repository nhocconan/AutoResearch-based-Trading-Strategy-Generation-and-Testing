#!/usr/bin/env python3
"""
12h_ema_bounce_volume_v1
Hypothesis: EMA bounce strategy with volume confirmation on 12h timeframe. 
Buy when price touches 12h EMA20 from below with volume surge, sell when price touches EMA20 from above with volume surge.
Works in both bull and bear markets by following the trend via EMA direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_bounce_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 on 12h data directly (no HTF needed for EMA)
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average for confirmation
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema20[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches EMA20 from below with volume confirmation
            if (close[i] >= ema20[i] and 
                close[i-1] < ema20[i-1] and  # Was below EMA previous bar
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: price touches EMA20 from above with volume confirmation
            elif (close[i] <= ema20[i] and 
                  close[i-1] > ema20[i-1] and  # Was above EMA previous bar
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals
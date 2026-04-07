#!/usr/bin/env python3
"""
6h_volume_acceleration_trend_12h_filter_v1
Hypothesis: Volume acceleration (current volume > 2x 10-period average) combined with 12h EMA trend filter on 6h timeframe.
In bull markets, enter long on volume acceleration + price above 12h EMA; in bear markets, enter short on volume acceleration + price below 12h EMA.
Volume acceleration filters for institutional participation, while EMA filter ensures trend alignment.
Works in both bull and bear by capturing momentum bursts with trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_acceleration_trend_12h_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume acceleration: current volume > 2x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_accel = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if required data not available
        if np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: volume acceleration dies OR price crosses below EMA
            if not vol_accel[i] or close[i] < ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: volume acceleration dies OR price crosses above EMA
            if not vol_accel[i] or close[i] > ema20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: volume acceleration + price above EMA (bullish momentum)
            if vol_accel[i] and close[i] > ema20_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: volume acceleration + price below EMA (bearish momentum)
            elif vol_accel[i] and close[i] < ema20_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
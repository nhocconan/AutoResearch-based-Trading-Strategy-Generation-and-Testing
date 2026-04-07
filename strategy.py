#!/usr/bin/env python3
"""
12h_williams_alligator_1w_trend_volume_v1
Hypothesis: Williams Alligator identifies trend direction on 1w. Price above/below Alligator jaws (13-period SMA) 
indicates trend bias. Enter long when price crosses above teeth (8-period SMA) with volume confirmation in uptrend,
enter short when price crosses below teeth in downtrend. Williams Alligator is trend-following but smooth, 
reducing whipsaw in choppy markets. Works in bull by following uptrend, in bear by following downtrend.
Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_williams_alligator_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Williams Alligator trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Williams Alligator components on 1w
    # Jaw: 13-period SMMA (smoothed moving average) - we'll use EMA as approximation
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).ewm(span=13, adjust=False).mean().values  # Jaw (13)
    teeth = pd.Series(close_1w).ewm(span=8, adjust=False).mean().values   # Teeth (8)
    lips = pd.Series(close_1w).ewm(span=5, adjust=False).mean().values    # Lips (5)
    
    # Align 1w Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Determine trend based on Alligator alignment
        # Uptrend: Lips > Teeth > Jaw (all aligned upward)
        # Downtrend: Lips < Teeth < Jaw (all aligned downward)
        is_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below teeth OR trend changes to downtrend
            if close[i] < teeth_aligned[i] or not is_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above teeth OR trend changes to uptrend
            if close[i] > teeth_aligned[i] or not is_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above teeth with volume and uptrend
            if (close[i] > teeth_aligned[i] and 
                close[i-1] <= teeth_aligned[i-1] and  # crossed above teeth
                vol_confirm and is_uptrend):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below teeth with volume and downtrend
            elif (close[i] < teeth_aligned[i] and 
                  close[i-1] >= teeth_aligned[i-1] and  # crossed below teeth
                  vol_confirm and is_downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals
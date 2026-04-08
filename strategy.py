#!/usr/bin/env python3
# 1d_alligator_1w_trend_volume
# Hypothesis: Williams Alligator on 1d filtered by 1w trend and volume confirmation on 1d.
# Long when price > Alligator Teeth (green line) with price > Alligator Jaw (blue line) and volume > 1.5x average.
# Short when price < Alligator Teeth (red line) with price < Alligator Jaw (blue line) and volume > 1.5x average.
# Designed to capture trending moves while avoiding choppy markets. Target: 20-40 trades/year (~80-160 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_alligator_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Williams Alligator lines (13, 8, 5 periods with shifts 8, 5, 3)
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align weekly Alligator lines to daily
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate average volume for confirmation (13-period)
    avg_volume = pd.Series(volume).rolling(window=13, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Alligator Teeth OR trend turns against us
            if (close[i] < teeth_aligned[i]) or (close[i] < jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Alligator Teeth OR trend turns against us
            if (close[i] > teeth_aligned[i]) or (close[i] > jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price > Alligator Teeth and > Alligator Jaw with volume confirmation
            if (close[i] > teeth_aligned[i]) and (close[i] > jaw_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price < Alligator Teeth and < Alligator Jaw with volume confirmation
            elif (close[i] < teeth_aligned[i]) and (close[i] < jaw_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals
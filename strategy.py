#!/usr/bin/env python3
"""
6h Volume Spike Reversal + 12h Trend Filter v1
Hypothesis: Volume spikes (>2.5x average) at Bollinger Band extremes (20,2) signal exhaustion reversals.
Filtered by 12h EMA(50) trend to avoid counter-trend trades. Works in bull/bear by capturing mean reversion
in overextended moves during volatile periods, with volume confirming institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_spike_reversal_12h_trend_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    # Volume filter (>2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle Bollinger Band or trend reverses
            if close[i] >= sma_20[i] or close[i] <= ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle Bollinger Band or trend reverses
            if close[i] <= sma_20[i] or close[i] >= ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long reversal at lower BB with volume spike and against trend
            if (close[i] <= lower_bb[i] and 
                vol_spike[i] and 
                close[i] < ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short reversal at upper BB with volume spike and against trend
            elif (close[i] >= upper_bb[i] and 
                  vol_spike[i] and 
                  close[i] > ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
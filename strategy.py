#!/usr/bin/env python3
"""
12h Williams Alligator with 1d Trend Filter and Volume Spike
Hypothesis: The Williams Alligator (three SMAs) identifies trends effectively.
When the Jaw, Teeth, and Lips are aligned (trending state), we take breakouts
in the direction of the trend confirmed by 1d EMA50 and volume spikes.
This captures strong trends while avoiding chop, working in both bull and bear markets.
Target: 12-37 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_williams_alligator_1d_trend_volume"
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
    
    # Williams Alligator (13, 8, 5 SMAs with offsets)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment: all three in order (trending state)
    # For uptrend: lips > teeth > jaw
    # For downtrend: lips < teeth < jaw
    alligator_up = (lips > teeth) & (teeth > jaw)
    alligator_down = (lips < teeth) & (teeth < jaw)
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    # 1d EMA50 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator turns against trend or price crosses below 1d EMA50
            if not alligator_up[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator turns against trend or price crosses above 1d EMA50
            if not alligator_down[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Alligator aligned up + price above 1d EMA50 + volume spike
            if (alligator_up[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Alligator aligned down + price below 1d EMA50 + volume spike
            elif (alligator_down[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
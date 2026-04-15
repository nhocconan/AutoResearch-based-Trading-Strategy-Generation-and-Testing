#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF context
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channels (20 periods)
    donch_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Calculate 12h volume average for spike detection
    vol_ma = pd.Series(df_12h['volume'].values).rolling(window=10, min_periods=10).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    # Volume spike: current volume > 1.8x 12-period average
    vol_spike = volume > (1.8 * vol_ma_aligned)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            continue
        
        # Long: Price breaks above 12h Donchian high + volume spike
        if close[i] > donch_high_aligned[i] and vol_spike[i]:
            signals[i] = 0.25
        
        # Short: Price breaks below 12h Donchian low + volume spike
        elif close[i] < donch_low_aligned[i] and vol_spike[i]:
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite breakout
        elif (close[i] < donch_low_aligned[i] and signals[i-1] > 0) or \
             (close[i] > donch_high_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_12h_Donchian20_Volume_Spike_Breakout"
timeframe = "6h"
leverage = 1.0
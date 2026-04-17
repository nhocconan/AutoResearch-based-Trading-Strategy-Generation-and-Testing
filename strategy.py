#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above 20-period high with volume > 1.3x average and price > 1d EMA50.
Short when price breaks below 20-period low with volume > 1.3x average and price < 1d EMA50.
Exit on opposite Donchian breakout or when price crosses 1d EMA50.
Uses 1d for EMA50 trend filter, 12h for price/volume/DN.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (current volume > 1.3x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        dn_high = highest_high[i]
        dn_low = lowest_low[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above 1d EMA50
            if price > dn_high and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below 1d EMA50
            elif price < dn_low and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR crosses below 1d EMA50
            if price < dn_low or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR crosses above 1d EMA50
            if price > dn_high or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0
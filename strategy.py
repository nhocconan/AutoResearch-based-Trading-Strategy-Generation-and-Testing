#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA25 for trend filter
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA25 to 4h timeframe
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema25_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: Donchian opposite break or volume fade
        exit_long = (close[i] < low_min[i]) or (not vol_confirm)
        exit_short = (close[i] > high_max[i]) or (not vol_confirm)
        
        if position == 1:  # Long position
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Price breaks above Donchian upper + 12h uptrend + volume
            if (close[i] > high_max[i]) and (close[i] > ema25_12h_aligned[i]) and vol_confirm:
                position = 1
                signals[i] = 0.30
            # Enter short: Price breaks below Donchian lower + 12h downtrend + volume
            elif (close[i] < low_min[i]) and (close[i] < ema25_12h_aligned[i]) and vol_confirm:
                position = -1
                signals[i] = -0.30
    
    return signals
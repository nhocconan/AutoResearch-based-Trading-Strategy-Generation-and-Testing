#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_1dTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (volume > 1.5 * 20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 for EMA and 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_30_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_30_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian upper + 1d trend up + volume spike
            if (close[i] > upper_channel and 
                close[i] > ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian lower + 1d trend down + volume spike
            elif (close[i] < lower_channel and 
                  close[i] < ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian lower or 1d trend turns down
            if (close[i] < lower_channel or 
                close[i] < ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian upper or 1d trend turns up
            if (close[i] > upper_channel or 
                close[i] > ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
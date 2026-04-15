#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Works in bull/bear: breakouts capture momentum, EMA50 filter avoids counter-trend trades.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Donchian20_EMA50_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need 20 bars for Donchian + 50 for EMA
    for i in range(50, n):
        # Skip if EMA data is NaN
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate 6h Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        highest_high = np.max(high[lookback_start:i+1])
        lowest_low = np.min(low[lookback_start:i+1])
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 19:
            vol_ma = np.mean(volume[i-19:i+1])
            vol_filter = volume[i] > 1.5 * vol_ma
        else:
            vol_filter = False
        
        # Long: breakout above Donchian high + above daily EMA50 + volume
        if (close[i] > highest_high and 
            close[i] > ema_50_1d_aligned[i] and 
            vol_filter):
            signals[i] = 0.25
            
        # Short: breakout below Donchian low + below daily EMA50 + volume
        elif (close[i] < lowest_low and 
              close[i] < ema_50_1d_aligned[i] and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals
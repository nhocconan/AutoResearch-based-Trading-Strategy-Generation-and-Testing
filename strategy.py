#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets.
# 1d EMA50 ensures we trade with the higher timeframe trend, reducing whipsaw.
# Volume spike (>1.5x 20-period EMA) confirms breakout strength.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.30) to balance return and drawdown.

name = "12h_Donchian20_1dEMA50_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period EMA of volume on 12h timeframe for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Calculate Donchian channels (20-period) using available data up to i
            lookback = min(20, i+1)
            highest_high = np.max(high[i-lookback+1:i+1])
            lowest_low = np.min(low[i-lookback+1:i+1])
            
            # Long: price breaks above Donchian high + above 1d EMA50 + volume confirmation
            if (close[i] > highest_high and 
                close[i] > ema_50_aligned[i] and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + below 1d EMA50 + volume confirmation
            elif (close[i] < lowest_low and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low (20) OR below 1d EMA50
            lookback = min(20, i+1)
            lowest_low = np.min(low[i-lookback+1:i+1])
            if (close[i] < lowest_low or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian high (20) OR above 1d EMA50
            lookback = min(20, i+1)
            highest_high = np.max(high[i-lookback+1:i+1])
            if (close[i] > highest_high or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
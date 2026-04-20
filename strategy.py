#!/usr/bin/env python3
# 4h_1d_Donchian20_Breakout_VolumeTrend
# Hypothesis: On 4h timeframe, trade breakouts from 20-period Donchian channels with volume confirmation and 1d EMA trend filter.
# The strategy captures trend continuation moves while avoiding false breakouts in choppy markets.
# Works in both bull and bear markets by aligning with the 1d trend direction.
# Uses volume spike (2x 20-period average) to confirm breakout strength.
# Targets 20-40 trades per year to minimize fee drag.

name = "4h_1d_Donchian20_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, and price above 1d EMA34 (uptrend)
            if (close[i] > high_max[i] and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, and price below 1d EMA34 (downtrend)
            elif (close[i] < low_min[i] and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal (below EMA34)
            if close[i] < low_min[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal (above EMA34)
            if close[i] > high_max[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
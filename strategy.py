#!/usr/bin/env python3
# 4H_DONCHIAN20_BREAKOUT_1D_VOLUME_FILTER
# Hypothesis: Donchian(20) breakouts capture momentum, filtered by 1D volume to avoid false breakouts.
# Works in bull markets (breakout continuations) and bear markets (sharp reversals after volatility).
# Target: 20-50 trades/year on 4h timeframe.

name = "4H_DONCHIAN20_BREAKOUT_1D_VOLUME_FILTER"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period SMA)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(vol_ma_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high with volume confirmation
            if (close[i] > high_20[i] and 
                volume[i] > vol_ma_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low with volume confirmation
            elif (close[i] < low_20[i] and 
                  volume[i] > vol_ma_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below 20-period low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above 20-period high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
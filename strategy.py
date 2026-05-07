# 12h_WickReversal_VolumeSpike_TopBottom
# Hypothesis: Rejections at daily high/low with volume spikes indicate exhaustion.
# In bull markets: buy dips at daily low with volume; in bear markets: sell rallies at daily high with volume.
# Works in both regimes by fading extreme rejections. Uses 1d high/low as dynamic support/resistance.
# Target: 15-30 trades/year on 12h timeframe to minimize fee drag.

#!/usr/bin/env python3
name = "12h_WickReversal_VolumeSpike_TopBottom"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align daily levels to 12h timeframe (wait for daily close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: rejection at daily low (long lower wick) + volume spike
            # Condition: low touches or goes below daily low, but close recovers above it
            # Plus strong volume indicating buying interest
            if (low[i] <= daily_low_aligned[i] and 
                close[i] > daily_low_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: rejection at daily high (long upper wick) + volume spike
            # Condition: high touches or goes above daily high, but close falls below it
            # Plus strong volume indicating selling pressure
            elif (high[i] >= daily_high_aligned[i] and 
                  close[i] < daily_high_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily low (mean reversion) or wick rejection against trend
            if close[i] <= daily_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily high (mean reversion) or wick rejection against trend
            if close[i] >= daily_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
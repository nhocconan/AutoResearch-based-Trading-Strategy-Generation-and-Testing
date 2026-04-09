#!/usr/bin/env python3
# 4h_donchian_12h_volume_v1
# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation for entry timing.
# Uses discrete sizing (±0.25) to minimize fee churn. Works in bull/bear by
# requiring volume spike to validate breakouts, reducing false signals in chop.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    # 12h volume MA(20)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h_aligned[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volume confirmation lost
            if low[i] <= low_min[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volume confirmation lost
            if high[i] >= high_max[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long: price breaks above Donchian high
                if high[i] > high_max[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif low[i] < low_min[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
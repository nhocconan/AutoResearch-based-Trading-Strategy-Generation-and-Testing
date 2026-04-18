#!/usr/bin/env python3
"""
1h_PriceChannel_Breakout_VolumeTrend
Hypothesis: Use 4h Donchian channels for trend direction and 1h price action with volume confirmation for precise entries. In bull markets, buy breakouts above 4h upper channel; in bear markets, sell breakdowns below 4h lower channel. Volume filter ensures institutional participation. Designed for low trade frequency (15-25/year) to minimize fee drift while capturing strong trends. Works in both bull and bear regimes by following the 4h trend.
"""

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
    
    # 4h Donchian channel (20-period) for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian channels on 4h
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    upper_20 = align_htf_to_ltf(prices, df_4h, high_20)
    lower_20 = align_htf_to_ltf(prices, df_4h, low_20)
    
    # 1h volume spike (>1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_20[i]
        lower = lower_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above 4h upper channel with volume spike
            if price > upper and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h lower channel with volume spike
            elif price < lower and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price returns to 4h midpoint or volume dies
            midpoint = (upper + lower) / 2
            if price < midpoint:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price returns to 4h midpoint or volume dies
            midpoint = (upper + lower) / 2
            if price > midpoint:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_PriceChannel_Breakout_VolumeTrend"
timeframe = "1h"
leverage = 1.0
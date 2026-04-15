# -*- coding: utf-8 -*-
# -*- mode: python -*-
#!/usr/bin/env python3

# HYPOTHESIS:
# 4h Donchian breakout with volume confirmation and volume regime filter.
# Long when price breaks above 20-period Donchian high + volume spike + volume in expansion regime.
# Short when price breaks below 20-period Donchian low + volume spike + volume in expansion regime.
# Exit when price crosses back through the Donchian midpoint.
# Uses 12h volume expansion regime filter to avoid chop and reduce whipsaws.
# Designed to work in both bull (trend continuation) and bear (mean reversion bounces) markets.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

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
    
    # Get 12h data for volume regime filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h volume moving average (20-period) for regime detection
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Volume expansion regime: current 12h volume > 1.5x 20-period average
    vol_expansion = df_12h['volume'].values > (1.5 * vol_ma_20_12h)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_12h, vol_expansion)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume spike: current volume > 2.0x 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(vol_expansion_aligned[i])):
            continue
        
        # Only trade in volume expansion regime (avoid chop)
        if not vol_expansion_aligned[i]:
            signals[i] = 0.0
            continue
            
        # Long: Donchian breakout above + volume spike
        if (close[i] > highest_20[i] and vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Donchian breakdown below + volume spike
        elif (close[i] < lowest_20[i] and vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back through Donchian midpoint
        elif (signals[i-1] > 0 and close[i] < donchian_mid[i]) or \
             (signals[i-1] < 0 and close[i] > donchian_mid[i]):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian20_Volume_Spike_Expansion_Filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
#100910 - 1d_Donchian20_Breakout_1wTrend_Volume
Hypothesis: Breakout at Donchian(20) channel with volume confirmation and 1w EMA trend filter on daily timeframe.
Targets 7-25 trades/year (30-100 total) to minimize fee drift. Uses daily breakout with weekly EMA40 for trend
and volume spike filter. Works in bull (breakouts with trend) and bear (mean reversion to middle band).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA40 for trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate Donchian channels (20-period) from daily data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max + low_min) / 2
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian band, above weekly EMA40, volume spike
        if (close[i] > high_max[i] and 
            close[i] > ema40_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian band, below weekly EMA40, volume spike
        elif (close[i] < low_min[i] and 
              close[i] < ema40_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Donchian middle band (mean reversion)
        elif position == 1 and close[i] < donchian_middle[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_middle[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0
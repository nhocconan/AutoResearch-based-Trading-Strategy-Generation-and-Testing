#!/usr/bin/env python3
"""
6h_volume_price_action_v1
Hypothesis: On 6h timeframe, combine volume spikes with price action at key levels.
In ranging markets, fade extremes on volume exhaustion; in trending markets, 
breakout on volume expansion. Uses 1w trend filter to avoid counter-trend trades.
Works in bull/bear by adapting to weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_price_action_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 6h indicators
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    # Price position relative to recent range (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    range_size = high_max - low_min
    # Avoid division by zero
    range_size = np.where(range_size == 0, 1e-10, range_size)
    price_pos = (close - low_min) / range_size  # 0=at low, 1=at high
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema20_6h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverses from high with volume spike OR weekly trend turns down
            if (price_pos[i] > 0.8 and vol_spike[i]) or close[i] < ema20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reverses from low with volume spike OR weekly trend turns up
            if (price_pos[i] < 0.2 and vol_spike[i]) or close[i] > ema20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade high in ranging market (weekly trend weak)
            if (price_pos[i] > 0.8 and vol_spike[i] and 
                abs(close[i] - ema20_6h[i]) / ema20_6h[i] < 0.05):  # near weekly EMA
                position = -1
                signals[i] = -0.25
            # Fade low in ranging market
            elif (price_pos[i] < 0.2 and vol_spike[i] and 
                  abs(close[i] - ema20_6h[i]) / ema20_6h[i] < 0.05):
                position = 1
                signals[i] = 0.25
            # Breakout high in trending market
            elif (price_pos[i] > 0.8 and vol_spike[i] and 
                  close[i] > ema20_6h[i]):
                position = 1
                signals[i] = 0.25
            # Breakout low in trending market
            elif (price_pos[i] < 0.2 and vol_spike[i] and 
                  close[i] < ema20_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
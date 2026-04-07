#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v1
Hypothesis: Donchian channel breakouts on 4h with 12h trend filter (EMA25) and volume confirmation.
Works in both bull and bear markets by only taking breakouts in direction of 12h trend.
Targets 20-50 trades/year (80-200 over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA25 for trend filter
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema25_12h_aligned[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns down
            if close[i] < low_roll[i] or close[i] < ema25_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns up
            if close[i] > high_roll[i] or close[i] > ema25_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian upper band in uptrend
            if (close[i] > high_roll[i] and 
                vol_confirm and 
                close[i] > ema25_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian lower band in downtrend
            elif (close[i] < low_roll[i] and 
                  vol_confirm and 
                  close[i] < ema25_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
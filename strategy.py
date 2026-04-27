#!/usr/bin/env python3
"""
#100842 - 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h timeframe strategy using Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike.
Designed to work in both bull and bear markets by capturing breakouts with trend confirmation and mean reversion to pivot.
Targets 12-37 trades/year to minimize fee drag on longer timeframe.
"""

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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r1 = close_1d + daily_range * 1.1 / 12
    daily_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align to 12h timeframe (previous day's levels for current period)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, daily_r1)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, daily_s1)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 1d EMA34, volume spike
        if (close[i] > camarilla_r1[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below 1d EMA34, volume spike
        elif (close[i] < camarilla_s1[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
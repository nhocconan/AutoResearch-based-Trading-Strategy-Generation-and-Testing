#!/usr/bin/env python3
"""
12h_1w_Donchian_55_WeeklyTrend_v1
Hypothesis: Use 1-week Donchian channels (55-period) as primary trend filter, with 12h price breakouts for entry and volume confirmation. 
Go long when 12h price breaks above weekly Donchian upper band, short when breaks below lower band. 
Requires volume > 2.0x 50-period average for confirmation to avoid false breakouts. 
Target: 15-25 trades/year by using weekly trend filter to significantly reduce noise. 
Works in bull markets via trend following and in bear via short signals aligned with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w Donchian channels (55-period)
    donch_len = 55
    upper_1w = np.full_like(high_1w, np.nan)
    lower_1w = np.full_like(low_1w, np.nan)
    
    if len(high_1w) >= donch_len:
        for i in range(donch_len, len(high_1w)):
            upper_1w[i] = np.max(high_1w[i-donch_len:i])
            lower_1w[i] = np.min(low_1w[i-donch_len:i])
    
    # Align Donchian channels to 12h timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 50
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + volume confirmation
            if close[i] > upper_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + volume confirmation
            elif close[i] < lower_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian lower
            if close[i] < lower_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian upper
            if close[i] > upper_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_55_WeeklyTrend_v1"
timeframe = "12h"
leverage = 1.0
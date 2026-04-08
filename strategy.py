#!/usr/bin/env python3
# 6h_1w_1d_volume_surge_breakout_v1
# Hypothesis: 6-hour price breaks Donchian(20) channel with volume surge (2x 20-period average) and weekly trend alignment.
# Long when price breaks above upper Donchian + volume surge + weekly uptrend.
# Short when price breaks below lower Donchian + volume surge + weekly downtrend.
# Weekly trend filter avoids counter-trend trades; volume surge confirms breakout strength.
# Designed for 15-35 trades/year on 6h to avoid fee drag. Works in bull/bear via volume confirmation and trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_volume_surge_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_avg[i] = vol_sum / 20.0
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Donchian and volume ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current volume > 2x 20-period average
        volume_surge = volume[i] > 2.0 * vol_avg[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below midpoint of Donchian channel
            mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] < mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above midpoint of Donchian channel
            mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if close[i] > mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian + volume surge + weekly uptrend
            if (close[i] > donchian_high[i] and 
                volume_surge and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian + volume surge + weekly downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_surge and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals
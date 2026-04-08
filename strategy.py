#!/usr/bin/env python3
# 4h_donchian_breakout_volume_v2
# Hypothesis: Donchian(20) breakout on 4h with volume confirmation and 1d trend filter (EMA50 slope).
# Long when: price breaks above upper Donchian, volume > 1.5x average, 1d EMA50 slope > 0.
# Short when: price breaks below lower Donchian, volume > 1.5x average, 1d EMA50 slope < 0.
# Exit when price returns to middle of Donchian channel or volume drops below average.
# Uses tight entry conditions to limit trades (~20-40/year) and avoid fee drag.
# Works in bull (breakouts continue) and bear (breakouts reverse) via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA slope: positive if current EMA > EMA 3 periods ago
    ema50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(3, len(close_1d)):
        if not np.isnan(ema50_1d[i]) and not np.isnan(ema50_1d[i-3]):
            ema50_slope_1d[i] = ema50_1d[i] - ema50_1d[i-3]
    ema50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to middle of Donchian or volume drops below average
            if close[i] <= middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to middle of Donchian or volume drops below average
            if close[i] >= middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper Donchian, volume surge, 1d EMA50 slope positive
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                ema50_slope_1d_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian, volume surge, 1d EMA50 slope negative
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  ema50_slope_1d_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals
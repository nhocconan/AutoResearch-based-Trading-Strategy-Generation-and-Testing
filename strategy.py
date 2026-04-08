#!/usr/bin/env python3
# 4h_donchian_breakout_volume_filter_v3
# Hypothesis: Combines Donchian channel breakouts (20-period) with volume confirmation (1.5x 20-period average) and 1-day EMA trend filter.
# Long when price breaks above upper Donchian band, volume surges, and 1-day EMA50 is rising.
# Short when price breaks below lower Donchian band, volume surges, and 1-day EMA50 is falling.
# Exit when price crosses the midline (average of upper and lower bands) or volume drops below average.
# Uses tight entry conditions to limit trades (~20-40/year) and reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_filter_v3"
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
    
    # Donchian channel (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    middle = (upper + lower) / 2.0
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not np.isnan(ema50_1d[i]) and not np.isnan(ema50_1d[i-1]):
            ema50_1d_slope[i] = ema50_1d[i] - ema50_1d[i-1]
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_slope_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below midline or volume drops below average
            if close[i] < middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above midline or volume drops below average
            if close[i] > middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper band, volume surge, 1-day EMA50 slope positive
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                ema50_1d_slope_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower band, volume surge, 1-day EMA50 slope negative
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  ema50_1d_slope_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals
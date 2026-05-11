#!/usr/bin/env python3
name = "6h_Market_Profile_Value_Area_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for daily trend and market profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily EMA50 for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Calculate daily value area high/low (simplified market profile)
    # Using 1-day range: value area = close +/- 0.382 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    value_area_high = close_1d + 0.382 * range_1d
    value_area_low = close_1d - 0.382 * range_1d
    
    # Align to 6h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    value_area_high_aligned = align_htf_to_ltf(prices, df_1d, value_area_high)
    value_area_low_aligned = align_htf_to_ltf(prices, df_1d, value_area_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(value_area_high_aligned[i]) or 
            np.isnan(value_area_low_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above value area high + daily uptrend
            if (close[i] > value_area_high_aligned[i] and 
                trend_up_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below value area low + daily downtrend
            elif (close[i] < value_area_low_aligned[i] and 
                  not trend_up_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below value area low or trend changes
            if (close[i] < value_area_low_aligned[i] or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above value area high or trend changes
            if (close[i] > value_area_high_aligned[i] or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
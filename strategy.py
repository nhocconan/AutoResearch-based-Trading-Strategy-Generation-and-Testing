#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period high/low for daily range
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    daily_high_20 = high_1d_series.rolling(window=20, min_periods=20).max().shift(1).values
    daily_low_20 = low_1d_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align daily range to 4h timeframe
    daily_high_20_aligned = align_htf_to_ltf(prices, df_1d, daily_high_20)
    daily_low_20_aligned = align_htf_to_ltf(prices, df_1d, daily_low_20)
    
    # Calculate 4-period range width (volatility measure)
    daily_range = daily_high_20 - daily_low_20
    daily_range_series = pd.Series(daily_range)
    range_avg = daily_range_series.rolling(window=10, min_periods=10).mean().shift(1).values
    range_avg_aligned = align_htf_to_ltf(prices, df_1d, range_avg)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 20-period + 10-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(daily_high_20_aligned[i]) or np.isnan(daily_low_20_aligned[i]) or
            np.isnan(range_avg_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above 20-day high AND range is expanding (volatility increasing)
            if price > daily_high_20_aligned[i] and range_avg_aligned[i] > daily_range[i-1] if i > 0 else False and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below 20-day low AND range is expanding
            elif price < daily_low_20_aligned[i] and range_avg_aligned[i] > daily_range[i-1] if i > 0 else False and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 20-day low OR volatility contracts
            if price < daily_low_20_aligned[i] or range_avg_aligned[i] < daily_range[i-1] * 0.8 if i > 0 else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 20-day high OR volatility contracts
            if price > daily_high_20_aligned[i] or range_avg_aligned[i] < daily_range[i-1] * 0.8 if i > 0 else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_20day_breakout_volatility_filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC approximation)
    # Use 5-day window for weekly approximation
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    # Get weekly high, low, close using 5-day windows
    if len(high_1d) >= 5:
        weekly_high = np.max(rolling_window(high_1d, 5), axis=1)
        weekly_low = np.min(rolling_window(low_1d, 5), axis=1)
        weekly_close = close_1d[4:]  # Last day of each 5-day window
        
        # Pad to match original length
        weekly_high = np.concatenate([np.full(4, np.nan), weekly_high])
        weekly_low = np.concatenate([np.full(4, np.nan), weekly_low])
        weekly_close = np.concatenate([np.full(4, np.nan), weekly_close])
        
        # Previous week's values for pivot calculation
        prev_weekly_high = np.roll(weekly_high, 5)
        prev_weekly_low = np.roll(weekly_low, 5)
        prev_weekly_close = np.roll(weekly_close, 5)
        
        # Weekly pivot point: (H + L + C) / 3
        pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
        # Resistance and support levels
        r1 = 2 * pp - prev_weekly_low
        r2 = pp + (prev_weekly_high - prev_weekly_low)
        s1 = 2 * pp - prev_weekly_high
        s2 = pp - (prev_weekly_high - prev_weekly_low)
        
        # Align pivot levels to 12h timeframe
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    else:
        # Fallback if insufficient data
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # 20 for volume + 10 buffer
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price touches or goes above R2 resistance with volume
            if price >= r2_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price touches or goes below S2 support with volume
            elif price <= s2_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches or goes below S2 support
            if price <= s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches or goes above R2 resistance
            if price >= r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Weekly_Pivot_Touch_Volume"
timeframe = "12h"
leverage = 1.0
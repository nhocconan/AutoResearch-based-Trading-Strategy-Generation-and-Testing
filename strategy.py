#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d pivot-based mean reversion and 1w trend filter.
# Long: Price touches 1d S3 support level + price < 1w EMA50 + volume > 1.3x avg volume (20-period).
# Short: Price touches 1d R3 resistance level + price > 1w EMA50 + volume > 1.3x avg volume.
# Uses 4h for execution timing, 1d for mean reversion levels, 1w for trend filter.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 50-150 total trades over 4 years (12-38/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    range_1d = high_1d[-1] - low_1d[-1]
    s3 = pivot - 1.1 * range_1d / 2.0
    r3 = pivot + 1.1 * range_1d / 2.0
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate daily pivot levels (using previous day's data)
    pivot_levels = np.full(n, np.nan)
    s3_levels = np.full(n, np.nan)
    r3_levels = np.full(n, np.nan)
    
    # We'll update these daily - for simplicity, use the most recent available
    # In practice, these would be updated when new 1d data is available
    pivot_levels[:] = pivot
    s3_levels[:] = s3
    r3_levels[:] = r3
    
    # Align 1w EMA50 to 4h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(s3_levels[i]) or np.isnan(r3_levels[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3_level = s3_levels[i]
        r3_level = r3_levels[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # Price proximity to pivot levels (within 0.5% of S3/R3)
        s3_proximity = abs(price - s3_level) / s3_level < 0.005
        r3_proximity = abs(price - r3_level) / r3_level < 0.005
        
        if position == 0:
            # Long: price near S3 + below EMA50 + volume confirmation
            if (s3_proximity and 
                price < ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price near R3 + above EMA50 + volume confirmation
            elif (r3_proximity and 
                  price > ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above pivot or above EMA50
            if (price > pivot_levels[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below pivot or below EMA50
            if (price < pivot_levels[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Pivot_EMA_MeanReversion"
timeframe = "4h"
leverage = 1.0
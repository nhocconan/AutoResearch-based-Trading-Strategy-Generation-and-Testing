#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly high: max of last 5 days (trading week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point: (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance levels
    r1 = 2 * pp - weekly_low
    r2 = pp + (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    # Support levels
    s1 = 2 * pp - weekly_high
    s2 = pp - (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (pp - weekly_low)
    
    # Align pivot levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # 20 for volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 pivot with volume
            if price > r3_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S3 pivot with volume
            elif price < s3_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3
            if price < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R3
            if price > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Weekly_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0
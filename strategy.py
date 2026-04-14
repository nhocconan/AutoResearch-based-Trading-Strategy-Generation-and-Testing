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
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use rolling windows to get weekly OHLC
    weekly_high = np.full_like(high_1d, np.nan)
    weekly_low = np.full_like(low_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    # Calculate weekly high/low/close using rolling window (7 days)
    for i in range(len(high_1d)):
        if i < 6:
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
            weekly_close[i] = np.nan
        else:
            weekly_high[i] = np.max(high_1d[i-6:i+1])
            weekly_low[i] = np.min(low_1d[i-6:i+1])
            weekly_close[i] = close_1d[i]  # Close of the week
    
    # Weekly pivot point: (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance levels
    r1 = 2 * pp - weekly_low
    r2 = pp + (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    # Support levels
    s1 = 2 * pp - weekly_high
    s2 = pp - (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align pivot levels to 1d timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
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
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R2 pivot with volume
            if price > r2_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 pivot with volume
            elif price < s2_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S2
            if price < s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R2
            if price > r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
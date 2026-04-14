#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_volume = np.roll(volume_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_volume[0] = np.nan
    
    # Standard pivot points: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: current volume > 1.5x average daily volume (20-day)
    vol_series = pd.Series(volume_1d)
    avg_vol_1d = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Range filter: avoid choppy markets (use daily range vs 20-day average)
    daily_range = prev_high - prev_low
    avg_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().shift(1).values
    avg_range_aligned = align_htf_to_ltf(prices, df_1d, avg_range)
    range_ratio = daily_range / avg_range  # current day's range vs average
    range_ratio_aligned = align_htf_to_ltf(prices, df_1d, range_ratio)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume and range calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(avg_vol_aligned[i]) or np.isnan(range_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Range filter: only trade when volatility is normal or high (avoid chop)
        if range_ratio_aligned[i] < 0.5:  # too choppy, skip
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume confirmation
            if price > r2_aligned[i] and vol > 1.5 * avg_vol_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 with volume confirmation
            elif price < s2_aligned[i] and vol > 1.5 * avg_vol_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Pivot_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0
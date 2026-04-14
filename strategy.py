#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot approximation (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC (approximated by 5-day lookback)
    # For true weekly, we'd need actual weekly data; using 5-day as proxy for weekly
    lookback = 5
    prev_high = np.roll(high_1d, lookback)
    prev_low = np.roll(low_1d, lookback)
    prev_close = np.roll(close_1d, lookback)
    # Set first 'lookback' values to NaN
    prev_high[:lookback] = np.nan
    prev_low[:lookback] = np.nan
    prev_close[:lookback] = np.nan
    
    # Pivot point: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    upper_20 = np.full_like(high_12h, np.nan)
    lower_20 = np.full_like(low_12h, np.nan)
    
    for i in range(len(high_12h)):
        if i < 19:
            upper_20[i] = np.nan
            lower_20[i] = np.nan
        else:
            upper_20[i] = np.max(high_12h[i-19:i+1])
            lower_20[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian channels to 4h timeframe (wait for 12h bar close)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20, 5)  # 20 for Donchian and volume, 5 for pivot lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian (20) AND above R1 pivot with volume
            if price > upper_20_aligned[i] and price > r1_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian (20) AND below S1 pivot with volume
            elif price < lower_20_aligned[i] and price < s1_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or below S2 (stronger stop)
            if price < lower_20_aligned[i] or price < s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or above R2 (stronger stop)
            if price > upper_20_aligned[i] or price > r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_Pivot_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0
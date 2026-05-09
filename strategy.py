#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Donchian20_Trend_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Weekly pivot from previous week (OHLC)
    # Calculate weekly bars: need to aggregate to weekly timeframe
    # We'll use a helper: get weekly data from prices
    # Since we only have 6h data, we'll calculate weekly pivot manually
    # Weekly high = max(high over last 28 bars (7 days * 4 bars per day))
    # Weekly low = min(low over last 28 bars)
    # Weekly close = close of bar 28 bars ago (end of previous week)
    # Weekly open = open of bar 28 bars ago
    weekly_lookback = 28  # 7 days * 4 six-hour bars per day
    
    # For each bar, weekly pivot based on previous week (not current forming week)
    # So we shift by weekly_lookback to get previous week's data
    weekly_high = pd.Series(high).rolling(window=weekly_lookback, min_periods=weekly_lookback).max().shift(weekly_lookback).values
    weekly_low = pd.Series(low).rolling(window=weekly_lookback, min_periods=weekly_lookback).min().shift(weekly_lookback).values
    weekly_close = np.roll(close, weekly_lookback)
    weekly_open = np.roll(prices['open'].values, weekly_lookback)
    
    # Handle initial NaN values
    for i in range(weekly_lookback):
        weekly_high[i] = high[i] if i < len(high) else np.nan
        weekly_low[i] = low[i] if i < len(low) else np.nan
        weekly_close[i] = close[i] if i < len(close) else np.nan
        weekly_open[i] = prices['open'].values[i] if i < len(prices['open'].values) else np.nan
    
    # Weekly pivot point and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly R1, S1, R2, S2, R3, S3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # 2. Donchian channel (20-period) on 6h
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 3. Daily trend filter: EMA 34 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4. Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, weekly_lookback + donchian_window)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 AND Donchian high AND daily uptrend AND volume
            if (price > weekly_r1[i] and 
                price > donchian_high[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below weekly S1 AND Donchian low AND daily downtrend AND volume
            elif (price < weekly_s1[i] and 
                  price < donchian_low[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to weekly pivot OR Donchian mid OR loses volume
            mid_point = (donchian_high[i] + donchian_low[i]) / 2.0
            if (price < weekly_pivot[i] or 
                price < mid_point or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot OR Donchian mid OR loses volume
            mid_point = (donchian_high[i] + donchian_low[i]) / 2.0
            if (price > weekly_pivot[i] or 
                price > mid_point or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
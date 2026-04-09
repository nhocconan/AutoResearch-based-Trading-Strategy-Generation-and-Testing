#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_volume_v1
# Hypothesis: 6h strategy using weekly Donchian breakouts filtered by 1d Camarilla pivot levels and volume confirmation. Enters long when price breaks above weekly Donchian high (20-period) with price > 1d S3 level and volume > 1.5x 20-period average. Enters short when price breaks below weekly Donchian low with price < 1d R3 level and volume confirmation. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 20-50 trades/year) to capture strong momentum moves aligned with intraday pivot structure, working in both bull and bear markets by requiring confluence of breakout, pivot alignment, and volume.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian Channels: upper = max(high, period), lower = min(low, period)"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float), np.full_like(low, np.nan, dtype=float)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    if len(high) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    # Typical price
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    pp = typical_price
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r3, r2, r1, pp, s1, s2, s3, s4

name = "6h_weekly_pivot_donchian_breakout_volume_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Donchian Channels (20-period on 6h data ≈ 5 trading days)
    donchian_high, donchian_low = calculate_donchian_channels(high, low, 20)
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d, _, _, _, _, _, s3_1d, _ = calculate_camarilla_pivots(high_1d, low_1d, close_1d)
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume and above 1d S3
            if (close[i] > donchian_high[i] and volume_confirmed and 
                close[i] > s3_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume and below 1d R3
            elif (close[i] < donchian_low[i] and volume_confirmed and 
                  close[i] < r3_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
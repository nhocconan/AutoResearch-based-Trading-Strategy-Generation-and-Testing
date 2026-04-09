#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Donchian captures breakouts; 1d weekly pivot provides HTF bias from higher timeframe structure
# Volume ensures breakout authenticity; discrete sizing 0.25 limits drawdown
# Works in bull/bear: pivot bias adapts to regime, breakouts work in both directions
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing

name = "6h_1d_pivot_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot points (using prior week's high/low/close)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll use rolling window of 5 trading days (approx 1 week) to get prior week's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling weekly high/low/close (5-day lookback for prior week)
    roll_high_5 = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # shift(1) for prior week
    roll_low_5 = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    roll_close_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (roll_high_5 + roll_low_5 + roll_close_5) / 3.0
    
    # Weekly support/resistance levels
    weekly_r1 = 2 * weekly_pivot - roll_low_5
    weekly_s1 = 2 * weekly_pivot - roll_high_5
    weekly_r2 = weekly_pivot + (roll_high_5 - roll_low_5)
    weekly_s2 = weekly_pivot - (roll_high_5 - roll_low_5)
    weekly_r3 = roll_high_5 + 2 * (weekly_pivot - roll_low_5)
    weekly_s3 = roll_low_5 - 2 * (roll_high_5 - weekly_pivot)
    
    # Align 1d weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < weekly S3 (strong bearish reversal)
            if close[i] < donchian_low[i] or close[i] < weekly_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > weekly R3 (strong bullish reversal)
            if close[i] > donchian_high[i] or close[i] > weekly_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + weekly pivot filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > weekly R3 (bullish breakout above resistance)
                if close[i] > donchian_high[i] and close[i] > weekly_r3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < weekly S3 (bearish breakdown below support)
                elif close[i] < donchian_low[i] and close[i] < weekly_s3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
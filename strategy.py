#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# Uses weekly Camarilla pivots (from Monday open) to determine trend direction:
# - Price above weekly R3: bullish bias (look for long breakouts)
# - Price below weekly S3: bearish bias (look for short breakouts)
# Entry: 6h Donchian(20) breakout in direction of weekly bias + volume spike (>2x 20-period avg)
# Exit: opposite Donchian breakout or loss of weekly bias
# Designed to capture strong trends with low frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once (for Camarilla pivots)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from previous week's OHLC
    # Based on weekly open, high, low, close
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla levels: R3/S3 and R4/S4
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    # S4 = close - (high - low) * 1.1
    rng = weekly_high - weekly_low
    r3 = weekly_close + rng * 1.1 / 2.0
    s3 = weekly_close - rng * 1.1 / 2.0
    r4 = weekly_close + rng * 1.1
    s4 = weekly_close - rng * 1.1
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly bias based on price relative to S3/R3
        # Bullish bias: price > weekly R3
        # Bearish bias: price < weekly S3
        # Neutral: between S3 and R3 (no new entries)
        bullish_bias = close[i] > r3_aligned[i]
        bearish_bias = close[i] < s3_aligned[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + bullish bias + volume spike
            if (close[i] > donchian_high[i] and 
                bullish_bias and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + bearish bias + volume spike
            elif (close[i] < donchian_low[i] and 
                  bearish_bias and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakout down OR loss of bullish bias
            if (close[i] < donchian_low[i] or not bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up OR loss of bearish bias
            if (close[i] > donchian_high[i] or not bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
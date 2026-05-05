#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + 1d Donchian Breakout
# Long when: BB(20,2) width < 20th percentile (squeeze) AND price breaks above 1d Donchian(20) high
# Short when: BB squeeze AND price breaks below 1d Donchian(20) low
# Exit when: BB width > 50th percentile (squeeze ends) OR opposite Donchian break
# Uses 6h for squeeze detection (low volatility precursors) and 1d Donchian for directional breakout
# Works in bull/bear: squeeze breaks often precede strong moves in either direction
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "6h_BBSqueeze_1dDonchianBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Donchian calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 6h Bollinger Bands (20,2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2 * dev
    lower_band = basis - 2 * dev
    bb_width = (upper_band - lower_band) / basis * 100  # Percentage width
    
    # Calculate 6h BB width percentiles for squeeze detection (using 50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=20).rank(pct=True).values * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_squeeze = bb_width_pct[i] < 20  # BB width in lower 20th percentile = squeeze
        bb_squeeze_end = bb_width_pct[i] > 50  # BB width above median = squeeze ending
        
        if position == 0:
            # Long: BB squeeze + price breaks above 1d Donchian high
            if bb_squeeze and close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + price breaks below 1d Donchian low
            elif bb_squeeze and close[i] < donch_low_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Squeeze ends OR price breaks below 1d Donchian low (reverse signal)
            if bb_squeeze_end or close[i] < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Squeeze ends OR price breaks above 1d Donchian high (reverse signal)
            if bb_squeeze_end or close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
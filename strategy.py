#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Bollinger Band squeeze (BB width < 20th percentile) as regime filter,
combined with 1w Donchian channel breakout (20-period) for entry direction.
Long when price breaks above weekly Donchian high during low volatility regime.
Short when price breaks below weekly Donchian low during low volatility regime.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
The Bollinger squeeze identifies periods of low volatility that often precede explosive moves,
while weekly Donchian breaks capture the direction of the ensuing volatility expansion.
Works in both bull and bear markets by trading breakouts in the direction of the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Bollinger Band width regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for Donchian channel breakout
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma_20 + (bb_std * bb_std_dev)
    lower_bb = ma_20 - (bb_std * bb_std_dev)
    bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    
    # Calculate 1d BB width percentile rank (20-period lookback)
    # Low volatility regime: BB width < 20th percentile
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    low_volatility_regime = bb_width_percentile < 0.20  # Bottom 20% = squeeze
    
    # Calculate weekly Donchian channel (20-period)
    donch_period = 20
    donch_high = pd.Series(high_1w).rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = pd.Series(low_1w).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align all to 6h
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility_regime)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for BB width percentile and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(low_volatility_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high during low volatility regime
            if (close[i] > donch_high_aligned[i] and 
                low_volatility_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low during low volatility regime
            elif (close[i] < donch_low_aligned[i] and 
                  low_volatility_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian midline
            donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
            if close[i] < donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian midline
            donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
            if close[i] > donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dBBSqueeze_1wDonchian20_Breakout"
timeframe = "6h"
leverage = 1.0
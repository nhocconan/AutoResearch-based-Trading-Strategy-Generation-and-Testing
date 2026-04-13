#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: Weekly Donchian channels provide strong trend-based support/resistance.
Breakouts above weekly high or below weekly low with volume confirmation capture institutional moves.
The daily EMA50 filter ensures trades align with the intermediate trend, reducing whipsaws.
This structure works in bull markets (continuation breakouts) and bear markets (fade at resistance)
by using price action confirmation rather than pure breakout logic. Targets 15-30 trades/year.
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
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily
    donchian_high_20w = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_20w = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_20w[i]) or np.isnan(donchian_low_20w[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above weekly Donchian high with volume expansion
        # 2. Must be above daily EMA50 for trend alignment
        breakout_long = (close[i] > donchian_high_20w[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_50_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below weekly Donchian low with volume expansion
        # 2. Must be below daily EMA50 for trend alignment
        breakdown_short = (close[i] < donchian_low_20w[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_50_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0
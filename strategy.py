#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: 12-hour Donchian channel breakouts with volume confirmation and daily trend filter capture institutional moves.
Long when price breaks above 20-period Donchian high with volume expansion and above daily EMA50.
Short when price breaks below 20-period Donchian low with volume expansion and below daily EMA50.
This structure works in bull markets (continuation breakouts) and bear markets (mean reversion at extremes)
by requiring volume confirmation and trend alignment, reducing false breakouts. Targets 12-37 trades/year.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above 20-period Donchian high with volume expansion
        # 2. Must be above daily EMA50 for trend alignment
        breakout_long = (close[i] > high_max_20[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_50_1d_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below 20-period Donchian low with volume expansion
        # 2. Must be below daily EMA50 for trend alignment
        breakdown_short = (close[i] < low_min_20[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_50_1d_aligned[i])
        
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

name = "12h_1d_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
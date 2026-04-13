#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance levels.
Price breaking above/below these levels with volume expansion indicates institutional participation.
Combined with daily trend filter (EMA50) to avoid counter-trend trades. Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formulas: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    resistance1 = close_1d + (range_1d * 1.1 / 12)
    resistance2 = close_1d + (range_1d * 1.1 / 6)
    resistance3 = close_1d + (range_1d * 1.1 / 4)
    resistance4 = close_1d + (range_1d * 1.1 / 2)
    
    support1 = close_1d - (range_1d * 1.1 / 12)
    support2 = close_1d - (range_1d * 1.1 / 6)
    support3 = close_1d - (range_1d * 1.1 / 4)
    support4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align all levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    resistance1_aligned = align_htf_to_ltf(prices, df_1d, resistance1)
    resistance2_aligned = align_htf_to_ltf(prices, df_1d, resistance2)
    resistance3_aligned = align_htf_to_ltf(prices, df_1d, resistance3)
    resistance4_aligned = align_htf_to_ltf(prices, df_1d, resistance4)
    support1_aligned = align_htf_to_ltf(prices, df_1d, support1)
    support2_aligned = align_htf_to_ltf(prices, df_1d, support2)
    support3_aligned = align_htf_to_ltf(prices, df_1d, support3)
    support4_aligned = align_htf_to_ltf(prices, df_1d, support4)
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(resistance1_aligned[i]) or 
            np.isnan(support1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above resistance3 (strong resistance level)
        # 2. Price above daily EMA50 (1d trend filter)
        # 3. Volume expansion
        breakout_long = close[i] > resistance3_aligned[i]
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        long_condition = breakout_long and price_above_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. Price breaks below support3 (strong support level)
        # 2. Price below daily EMA50 (1d trend filter)
        # 3. Volume expansion
        breakdown_short = close[i] < support3_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        short_condition = breakdown_short and price_below_ema and volume_expansion[i]
        
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

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
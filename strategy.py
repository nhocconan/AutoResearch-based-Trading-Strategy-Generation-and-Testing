#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Bollinger Band squeeze (low volatility) and 4-hour Donchian channel breakout.
# In low volatility regimes (BB width < 20th percentile), price tends to mean-revert to the Bollinger mid-band (20 SMA).
# Enters long when price crosses above the 20 SMA in low-volatility regime, short when below.
# Uses 4-hour Donchian breakout as confirmation: only take longs when price > 4h Donchian upper, shorts when < 4h Donchian lower.
# Exits when volatility regime shifts to high volatility or price reverts to the 20 SMA.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_BB_Squeeze_Donchian_Confirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: low volatility when BB width < 20th percentile
    bb_width_percentile = bb_width.rolling(window=100, min_periods=100).quantile(0.2)
    bb_squeeze = bb_width < bb_width_percentile
    bb_squeeze_values = bb_squeeze.values
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze_values)
    
    # Bollinger mid-band (20 SMA) for mean reversion target
    bb_mid = sma_20.values
    bb_mid_aligned = align_htf_to_ltf(prices, df_1d, bb_mid)
    
    # Price position relative to BB mid-band
    price_above_mid = close > bb_mid_aligned
    price_below_mid = close < bb_mid_aligned
    
    # 4-hour Donchian channel (20-period) for breakout confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    donchian_upper = high_4h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_4h.rolling(window=20, min_periods=20).min()
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_values)
    
    # Breakout conditions: price > 4h Donchian upper (long), price < 4h Donchian lower (short)
    price_above_donchian_upper = close > donchian_upper_aligned
    price_below_donchian_lower = close < donchian_lower_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_squeeze_aligned[i]) or
            np.isnan(bb_mid_aligned[i]) or
            np.isnan(price_above_mid[i]) or np.isnan(price_below_mid[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(price_above_donchian_upper[i]) or np.isnan(price_below_donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility (BB squeeze) + price above BB mid + price > 4h Donchian upper
            if bb_squeeze_aligned[i] and price_above_mid[i] and price_above_donchian_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility (BB squeeze) + price below BB mid + price < 4h Donchian lower
            elif bb_squeeze_aligned[i] and price_below_mid[i] and price_below_donchian_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR price crosses below BB mid
            if (not bb_squeeze_aligned[i]) or (not price_above_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR price crosses above BB mid
            if (not bb_squeeze_aligned[i]) or (not price_below_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
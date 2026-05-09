#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly Bollinger Band squeeze (low volatility) and daily Donchian channel breakout.
# In low volatility regimes (BB width < 20th percentile on weekly), price tends to breakout in the direction of the daily trend.
# Enters long when price crosses above the daily Donchian upper in low-volatility regime, short when below daily Donchian lower.
# Exits when volatility regime shifts to high volatility or price reverses to the daily Donchian midpoint.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_WeeklyBB_Squeeze_DailyDonchian_Breakout"
timeframe = "1d"
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
    
    # Calculate weekly Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    sma_20w = close_1w.rolling(window=20, min_periods=20).mean()
    std_20w = close_1w.rolling(window=20, min_periods=20).std()
    upper_bbw = sma_20w + 2 * std_20w
    lower_bbw = sma_20w - 2 * std_20w
    bb_widthw = upper_bbw - lower_bbw
    
    # Bollinger Band squeeze: low volatility when BB width < 20th percentile
    bb_width_percentilew = bb_widthw.rolling(window=50, min_periods=50).quantile(0.2)
    bb_squeezew = bb_widthw < bb_width_percentilew
    bb_squeeze_valuesw = bb_squeezew.values
    bb_squeeze_alignedw = align_htf_to_ltf(prices, df_1w, bb_squeeze_valuesw)
    
    # Calculate daily Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    donchian_upper = high_1d.rolling(window=20, min_periods=20).max()
    donchian_lower = low_1d.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_mid_values = donchian_mid.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_values)
    
    # Breakout conditions: price > daily Donchian upper (long), price < daily Donchian lower (short)
    price_above_donchian_upper = close > donchian_upper_aligned
    price_below_donchian_lower = close < donchian_lower_aligned
    price_above_donchian_mid = close > donchian_mid_aligned
    price_below_donchian_mid = close < donchian_mid_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_squeeze_alignedw[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(price_above_donchian_upper[i]) or np.isnan(price_below_donchian_lower[i]) or
            np.isnan(price_above_donchian_mid[i]) or np.isnan(price_below_donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility (weekly BB squeeze) + price > daily Donchian upper
            if bb_squeeze_alignedw[i] and price_above_donchian_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility (weekly BB squeeze) + price < daily Donchian lower
            elif bb_squeeze_alignedw[i] and price_below_donchian_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility shifts to high OR price crosses below daily Donchian mid
            if (not bb_squeeze_alignedw[i]) or (not price_above_donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility shifts to high OR price crosses above daily Donchian mid
            if (not bb_squeeze_alignedw[i]) or (not price_below_donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Choppiness Index (14) regime filter + 1-day Donchian (20) breakout
# Only trade when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets
# Long when price breaks above 1-day Donchian upper channel in trending regime
# Short when price breaks below 1-day Donchian lower channel in trending regime
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Uses daily timeframe for regime and structure, 12h for execution
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dChop_Trend_Donchian20_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1-day Choppiness Index (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over 14 periods
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero when hh == ll
    range_hl = hh - ll
    chop = 100 * np.log10(atr_sum / range_hl.replace(0, np.nan)) / np.log10(14)
    chop_values = chop.values
    
    # Calculate 1-day Donchian Channel (20-period high/low)
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(chop_aligned[i]) or np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending regime (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Long breakout: price breaks above upper Donchian
            if close[i] > upper_donchian[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian
            elif close[i] < lower_donchian[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
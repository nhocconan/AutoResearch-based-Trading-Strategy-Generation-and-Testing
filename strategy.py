#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Long when price breaks above 20-period Donchian high, 1d weekly pivot shows bullish bias (price > weekly pivot), and volume > average
# Short when price breaks below 20-period Donchian low, 1d weekly pivot shows bearish bias (price < weekly pivot), and volume > average
# Weekly pivot calculated from prior week's OHLC: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
# Uses volume confirmation to avoid false breakouts
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets

name = "6h_donchian20_1d_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC
    # Need at least 5 days for a full week
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using prior week's data (shift by 5 days)
    if len(high_1d) >= 5:
        # Rolling window of 5 days for weekly OHLC
        week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point: P = (WeekHigh + WeekLow + WeekClose) / 3
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        # Resistance 1: R1 = 2*P - WeekLow
        r1 = 2 * weekly_pivot - week_low
        # Support 1: S1 = 2*P - WeekHigh
        s1 = 2 * weekly_pivot - week_high
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    else:
        # Not enough data for weekly pivot
        weekly_pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Support 1 or price < weekly pivot (bearish bias)
            elif close[i] < s1_aligned[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Resistance 1 or price > weekly pivot (bullish bias)
            elif close[i] > r1_aligned[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long when price breaks above Donchian high and price > weekly pivot (bullish bias)
                if close[i] > high_max[i] and close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price breaks below Donchian low and price < weekly pivot (bearish bias)
                elif close[i] < low_min[i] and close[i] < weekly_pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals
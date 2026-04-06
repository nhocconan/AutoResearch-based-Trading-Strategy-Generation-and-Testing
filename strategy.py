#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter
# Long when price breaks above Donchian(20) high + weekly pivot bias bullish + volume confirmation
# Short when price breaks below Donchian(20) low + weekly pivot bias bearish + volume confirmation
# Weekly pivot calculated from 1d data: pivot = (weekly_high + weekly_low + weekly_close)/3
# Bias: bullish if current price > weekly pivot, bearish if < weekly pivot
# Uses volume > 1.3x average for confirmation to avoid false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk

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
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot from 1d data
    # Group into weeks (7 days) and calculate weekly high, low, close
    weeks = len(close_1d) // 7
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    for w in range(weeks):
        start = w * 7
        end = start + 7
        if end <= len(close_1d):
            week_high = np.max(high_1d[start:end])
            week_low = np.min(low_1d[start:end])
            week_close = close_1d[end-1]
            weekly_high[start:end] = week_high
            weekly_low[start:end] = week_low
            weekly_close[start:end] = week_close
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian channels (20-period) on 6h data
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i])):
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
            # Exit: price breaks below Donchian lower or weekly pivot turns bearish
            elif close[i] < donchian_lower[i] or close[i] < weekly_pivot_aligned[i]:
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
            # Exit: price breaks above Donchian upper or weekly pivot turns bullish
            elif close[i] > donchian_upper[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above Donchian upper + bullish pivot bias + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > weekly_pivot_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower + bearish pivot bias + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < weekly_pivot_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
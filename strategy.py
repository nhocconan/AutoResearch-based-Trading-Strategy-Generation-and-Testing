#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high, 1d price above weekly pivot (bullish bias), and volume > 1.5x average
# Short when price breaks below Donchian(20) low, 1d price below weekly pivot (bearish bias), and volume > 1.5x average
# Weekly pivot calculated from prior week's (Monday-Sunday) high, low, close
# Exit when price crosses Donchian midpoint or reverses with volume confirmation
# Target: 75-200 total trades over 4 years with controlled risk (max 0.25 position)

name = "6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot points (using Sunday as week end)
    # Pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll calculate this daily and forward-fill to 6h
    weekly_pivot = np.full_like(close_1d, np.nan)
    
    # For each day, determine if it's Sunday (week end)
    # Assuming data starts at some point, we'll use a rolling 7-day window
    # and assume the last day of the window is Sunday
    for i in range(6, len(close_1d)):  # Need 7 days for a week
        week_high = np.max(high_1d[i-6:i+1])
        week_low = np.min(low_1d[i-6:i+1])
        week_close = close_1d[i]
        weekly_pivot[i] = (week_high + week_low + week_close) / 3.0
    
    # Forward fill for days before first complete week
    for i in range(1, len(weekly_pivot)):
        if np.isnan(weekly_pivot[i]):
            weekly_pivot[i] = weekly_pivot[i-1]
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Donchian(20) channels
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or bearish pivot rejection
            elif close[i] < middle_20[i] or (close[i] < weekly_pivot_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or bullish pivot rejection
            elif close[i] > middle_20[i] or (close[i] > weekly_pivot_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above Donchian upper, price above weekly pivot (bullish bias), volume spike
            if (close[i] > upper_20[i] and 
                close[i] > weekly_pivot_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower, price below weekly pivot (bearish bias), volume spike
            elif (close[i] < lower_20[i] and 
                  close[i] < weekly_pivot_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter (EMA50) + volume confirmation
# Uses the 20-period high/low channel on 4h for breakout signals. Trades only in direction of
# 1d EMA50 trend to avoid counter-trend whipsaws. Volume > 1.5x median confirms breakout strength.
# Works in bull markets (long breakouts above EMA50) and bear markets (short breakdowns below EMA50).
# Target: 50-150 total trades over 4 years (12-38/year). Timeframe: 4h, HTF: 1d.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe (wait for completed 1d candle)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + above 1d EMA50 + volume confirmation
        if (close[i] > highest_high[i] and 
            close[i] > ema_50_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + below 1d EMA50 + volume confirmation
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_50_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout or price crosses 1d EMA50 in opposite direction
        elif position == 1 and (close[i] < lowest_low[i] or close[i] < ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or close[i] > ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA50Trend_Volume"
timeframe = "4h"
leverage = 1.0
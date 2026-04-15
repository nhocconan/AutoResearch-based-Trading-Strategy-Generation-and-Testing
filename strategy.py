#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Trades breakouts of the 20-day high/low only when aligned with the weekly trend (EMA50).
# Volume must exceed 1.5x the 20-period median to confirm breakout strength.
# Works in bull markets (long breakouts) and bear markets (short breakouts).
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on daily
    # Highest high of past 20 days (including current)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lowest low of past 20 days (including current)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on weekly
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1d timeframe (no shift needed as we use completed daily bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Align 1w EMA to 1d timeframe (waits for weekly close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 1.5x median of past 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_median[i])):
            continue
        
        # Long entry: price breaks above Donchian high + weekly uptrend + volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            close[i] > ema_50_aligned[i] and  # Price above weekly EMA = uptrend
            volume[i] > 1.5 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + weekly downtrend + volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < ema_50_aligned[i] and  # Price below weekly EMA = downtrend
              volume[i] > 1.5 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Donchian breakout or loss of weekly trend alignment
        elif position == 1 and (close[i] < donchian_low_aligned[i] or close[i] < ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_aligned[i] or close[i] > ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_20_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0
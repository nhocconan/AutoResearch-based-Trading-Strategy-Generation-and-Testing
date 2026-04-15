#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 20-period Donchian breakout with weekly EMA trend filter and volume confirmation
# Long when price breaks above 20-day high + weekly EMA(50) up + volume > 1.5x average
# Short when price breaks below 20-day low + weekly EMA(50) down + volume > 1.5x average
# Exit on opposite breakout or weekly EMA flip
# Target: 20-50 trades/year on daily timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels (using previous day's data to avoid look-ahead)
    # Highest high of last 20 days (excluding current day)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to daily timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate weekly EMA(50)
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: price breaks above 20-day high + weekly EMA up + volume confirmation
        if (close[i] > highest_20_aligned[i] and
            ema_50_aligned[i] > ema_50_aligned[i-1] and  # EMA rising
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 20-day low + weekly EMA down + volume confirmation
        elif (close[i] < lowest_20_aligned[i] and
              ema_50_aligned[i] < ema_50_aligned[i-1] and  # EMA falling
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or EMA trend change
        elif position == 1 and (close[i] < lowest_20_aligned[i] or ema_50_aligned[i] < ema_50_aligned[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_20_aligned[i] or ema_50_aligned[i] > ema_50_aligned[i-1]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0
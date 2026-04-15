#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and weekly trend filter
# Uses the 20-day high/low as breakout levels. Breakouts are taken only when
# volume is above average and the weekly trend (via 50-week EMA) aligns with the breakout direction.
# Works in bull markets (breakouts above 20-day high with weekly uptrend) and bear markets
# (breakouts below 20-day low with weekly downtrend). Target: 30-100 total trades over 4 years.
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day Donchian channels (highest high, lowest low over 20 days)
    # Using rolling window on daily data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above 20-day high + volume confirmation + weekly uptrend
        if (close[i] > highest_20[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 20-day low + volume confirmation + weekly downtrend
        elif (close[i] < lowest_20[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_50_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or price crosses back to weekly EMA (trend change)
        elif position == 1 and (close[i] < lowest_20[i] or close[i] < ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_20[i] or close[i] > ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0
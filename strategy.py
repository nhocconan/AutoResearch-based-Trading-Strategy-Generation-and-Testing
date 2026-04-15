#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation
# Uses weekly high/low as support/resistance. Breakouts above weekly high or below weekly low
# are traded with volume confirmation. Works in bull (breakouts up) and bear (breakouts down)
# markets. Weekly timeframe reduces noise and overtrading, targeting 30-100 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly Donchian channels (20-week high/low)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    weekly_high_20 = rolling_max(weekly_high, 20)
    weekly_low_20 = rolling_min(weekly_low, 20)
    
    # Align weekly channels to daily timeframe
    weekly_high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    weekly_low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)
    
    # Volume confirmation: current volume > 1.5x 20-day median volume
    volume_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_20_aligned[i]) or np.isnan(weekly_low_20_aligned[i]) or
            np.isnan(volume_median[i])):
            continue
        
        # Long entry: price breaks above weekly high + volume confirmation
        if (close[i] > weekly_high_20_aligned[i] and
            volume[i] > 1.5 * volume_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly low + volume confirmation
        elif (close[i] < weekly_low_20_aligned[i] and
              volume[i] > 1.5 * volume_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout (reversion to mean)
        elif position == 1 and close[i] < weekly_low_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > weekly_high_20_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
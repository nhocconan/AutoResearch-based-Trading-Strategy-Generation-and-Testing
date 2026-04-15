#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(10) breakout + volume confirmation + 1w EMA200 trend filter
# Uses daily price channel breakouts for trend capture, volume to confirm breakout strength,
# and weekly EMA200 to filter for long-term trend direction. Works in bull markets (longs)
# and bear markets (shorts) by only taking breakouts in the direction of the weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (10-period) on 1d
    donch_high_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Calculate EMA200 on 1w for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above weekly EMA200
        if (close[i] > donch_high_1d_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema200_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below weekly EMA200
        elif (close[i] < donch_low_1d_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema200_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < donch_low_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0
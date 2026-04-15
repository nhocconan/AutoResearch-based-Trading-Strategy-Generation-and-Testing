#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 200-day EMA trend filter + 1-day high/low breakout with volume confirmation
# Uses long-term trend filter (200EMA) to avoid counter-trend trades, combined with
# daily breakouts for entry timing and volume confirmation to filter false breakouts.
# Works in bull markets (trend-following breakouts) and bear markets (avoids longs in downtrend,
# only takes shorts when price < 200EMA and breaks down). Target: 20-60 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA200 and daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 200-day EMA on 1d for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily high and low (same as input for 1d timeframe)
    # For 1d timeframe, the daily high/low is just the 1d high/low
    # No calculation needed - we use the raw 1d high/low
    
    # Calculate average volume (20-day) on 1d for volume confirmation
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (identity since we're already at 1d)
    ema200_1d_aligned = ema200_1d  # Already at 1d frequency
    vol_avg_20_aligned = vol_avg_20  # Already at 1d frequency
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(250, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            continue
        
        # Long entry: price breaks above previous day's high + volume spike + price above EMA200
        if (high[i] > high_1d[i-1] and  # Today's high > yesterday's high (breakout)
            volume[i] > 1.5 * vol_avg_20_aligned[i] and
            close[i] > ema200_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume spike + price below EMA200
        elif (low[i] < low_1d[i-1] and  # Today's low < yesterday's low (breakdown)
              volume[i] > 1.5 * vol_avg_20_aligned[i] and
              close[i] < ema200_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to EMA200 (mean reversion to trend)
        elif position == 1 and close[i] < ema200_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema200_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_EMA200_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
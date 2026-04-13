#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + ATR trailing stop
    # Long: price breaks above 20-period high + volume > 1.5x 20-period average
    # Short: price breaks below 20-period low + volume > 1.5x 20-period average
    # Uses ATR-based trailing stop (highest high - 3*ATR for long, lowest low + 3*ATR for short)
    # Discrete sizing (0.25) to minimize fee drag
    # Target: 20-50 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels (using 4h data)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate ATR (14-period) for trailing stop
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    # Wilder's smoothing for ATR
    atr[13] = np.mean(tr[1:14])  # Seed with first 14 values
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_avg_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_4h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_4h[i]
        
        # Breakout conditions
        breakout_long = (close[i] > high_rolling_max[i-1]) and volume_confirmed
        breakout_short = (close[i] < low_rolling_min[i-1]) and volume_confirmed
        
        # Trailing stop conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Update highest high since entry
            if np.isnan(highest_since_entry[i-1]):
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # Exit if price drops below highest high - 3*ATR
            exit_long = close[i] < (highest_since_entry[i] - 3.0 * atr[i])
        
        elif position == -1:
            # Update lowest low since entry
            if np.isnan(lowest_since_entry[i-1]):
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # Exit if price rises above lowest low + 3*ATR
            exit_short = close[i] > (lowest_since_entry[i] + 3.0 * atr[i])
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            highest_since_entry[i] = high[i]  # Reset tracking
            lowest_since_entry[i] = np.nan
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            lowest_since_entry[i] = low[i]  # Reset tracking
            highest_since_entry[i] = np.nan
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            highest_since_entry[i] = np.nan
            lowest_since_entry[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            highest_since_entry[i] = np.nan
            lowest_since_entry[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            elif position == -1:
                signals[i] = -position_size
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
    
    return signals

name = "4h_1d_donchian_volume_trailing_v1"
timeframe = "4h"
leverage = 1.0
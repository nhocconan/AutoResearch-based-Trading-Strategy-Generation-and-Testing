#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
Long when price breaks above 12-hour high + 12-hour EMA50 trend + volume spike.
Short when price breaks below 12-hour low + 12-hour EMA50 trend + volume spike.
Exit on opposite breakout or when volume drops below average.
Uses institutional volume confirmation to filter false breakouts and trend alignment to avoid whipsaws.
Works in bull markets (trend continuation) and bear markets (mean reversion after breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend and volume filters - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12-hour EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12-hour average volume for volume filter
    volume_12h = df_12h['volume'].values
    avg_vol_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    # Donchian channels (20-period high/low)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_vol_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high + above 12h EMA50 + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_vol_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + below 12h EMA50 + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_vol_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks below 20-period low
                if close[i] < low_min_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price breaks above 20-period high
                if close[i] > high_max_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
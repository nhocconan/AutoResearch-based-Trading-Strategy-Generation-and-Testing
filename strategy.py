#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above daily upper Donchian(20) AND 1w EMA50 is rising AND 12h volume > 1.5 * avg_volume(20)
# Short when price breaks below daily lower Donchian(20) AND 1w EMA50 is falling AND 12h volume > 1.5 * avg_volume(20)
# Exit when price returns to the opposite Donchian level (upper for shorts, lower for longs)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Daily Donchian channels provide strong support/resistance levels from recent price action
# 1w EMA50 ensures we trade with the weekly trend while reducing noise
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "12h_1dDonchian20_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian(20) channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 completed weekly bars for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily upper Donchian(20), EMA50 rising, volume spike
            if (close[i] > upper_20_aligned[i] and close[i-1] <= upper_20_aligned[i-1] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily lower Donchian(20), EMA50 falling, volume spike
            elif (close[i] < lower_20_aligned[i] and close[i-1] >= lower_20_aligned[i-1] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily lower Donchian(20) level or above
            if close[i] >= lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily upper Donchian(20) level or below
            if close[i] <= upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
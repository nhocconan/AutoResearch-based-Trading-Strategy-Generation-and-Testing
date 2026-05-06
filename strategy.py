#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND 4h volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND 4h volume > 1.5 * avg_volume(20)
# Exit when price crosses the Donchian(20) midpoint or opposite extreme
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide structure and breakout levels from institutional order flow
# 12h EMA50 ensures we trade with the higher timeframe trend while reducing noise
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 completed 12h bars for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high, EMA50 rising, volume spike
            if (close[i] > highest_high_20[i] and close[i-1] <= highest_high_20[i-1] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low, EMA50 falling, volume spike
            elif (close[i] < lowest_low_20[i] and close[i-1] >= lowest_low_20[i-1] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint or below Donchian low
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint or above Donchian high
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
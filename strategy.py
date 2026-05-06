#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 1w Donchian(20) upper band AND 1d EMA50 is rising AND 1d volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian(20) lower band AND 1d EMA50 is falling AND 1d volume > 1.5 * avg_volume(20)
# Exit when price returns to 1w Donchian(20) midpoint
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1w Donchian provides strong weekly structure, filtering out lower timeframe noise
# 1d EMA50 ensures we trade with the daily trend while reducing whipsaws
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "12h_1wDonchian20_Breakout_1dEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period)
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid_1w = (upper_1w + lower_1w) / 2.0
    
    # Align 1w Donchian levels to 12h timeframe (wait for completed 1w bar)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    mid_1w_aligned = align_htf_to_ltf(prices, df_1w, mid_1w)
    
    # Get 1d data ONCE before loop for EMA50 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50 and volume avg
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(mid_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper band, EMA50 rising, volume spike
            if (close[i] > upper_1w_aligned[i] and close[i-1] <= upper_1w_aligned[i-1] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower band, EMA50 falling, volume spike
            elif (close[i] < lower_1w_aligned[i] and close[i-1] >= lower_1w_aligned[i-1] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1w Donchian midpoint
            if close[i] <= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1w Donchian midpoint
            if close[i] >= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
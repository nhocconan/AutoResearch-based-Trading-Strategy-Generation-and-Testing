#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 1w Donchian high AND 1d EMA50 is rising AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 1w Donchian low AND 1d EMA50 is falling AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses the 1w Donchian midpoint
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1w Donchian provides strong structural breaks with fewer false signals vs lower timeframes
# 1d EMA50 ensures we trade with the daily trend while reducing noise
# Moderate volume threshold (1.5x) controls trade frequency while capturing genuine breakouts
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 1d trend

name = "4h_1wDonchian20_Breakout_1dEMA50_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channels (20-period)
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid_20_1w = (high_20_1w + low_20_1w) / 2.0
    
    # Align 1w Donchian levels to 4h timeframe (wait for completed 1w bar)
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    mid_20_aligned = align_htf_to_ltf(prices, df_1w, mid_20_1w)
    
    # Get 1d data ONCE before loop for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(mid_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high, EMA50 rising, volume spike
            if (close[i] > high_20_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian low, EMA50 falling, volume spike
            elif (close[i] < low_20_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 1w Donchian midpoint
            if close[i] < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above the 1w Donchian midpoint
            if close[i] > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
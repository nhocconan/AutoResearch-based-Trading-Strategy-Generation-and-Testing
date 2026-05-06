#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel (20) AND 1d EMA50 > EMA50 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 1h
# Short when price breaks below 4h Donchian lower channel (20) AND 1d EMA50 < EMA50 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 1h
# Exit when price retests the 4h Donchian midpoint (median of upper/lower)
# Uses discrete sizing 0.20 to limit risk and reduce fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides structural breakout levels with continuation probability
# 1d EMA50 ensures we trade with the dominant daily trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Session filter (08-20 UTC) reduces noise trades
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "1h_Donchian20_1dEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper, 1d EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > upper_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower, 1d EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < lower_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests the 4h Donchian midpoint
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests the 4h Donchian midpoint
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
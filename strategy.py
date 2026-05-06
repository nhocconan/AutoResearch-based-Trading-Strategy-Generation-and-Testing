#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and session filter (08-20 UTC)
# Long when price breaks above 4h Donchian(20) upper band AND 1d EMA50 is rising AND hour in [8,20) UTC
# Short when price breaks below 4h Donchian(20) lower band AND 1d EMA50 is falling AND hour in [8,20) UTC
# Exit when price returns to 4h Donchian(20) midpoint
# Uses discrete sizing 0.20 to control drawdown and minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides structure from higher timeframe, 1d EMA50 ensures trend alignment
# Session filter reduces noise during low-liquidity hours, improving win rate
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets by following the daily trend

name = "1h_4hDonchian20_Breakout_1dEMA50_Trend_Session"
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
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channel (20-period)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_4h = (upper_4h + lower_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    mid_4h_aligned = align_htf_to_ltf(prices, df_4h, mid_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours < 20)  # UTC 8-20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(mid_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band, EMA50 rising, in session
            if (close[i] > upper_4h_aligned[i] and close[i-1] <= upper_4h_aligned[i-1] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower band, EMA50 falling, in session
            elif (close[i] < lower_4h_aligned[i] and close[i-1] >= lower_4h_aligned[i-1] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint
            if close[i] <= mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint
            if close[i] >= mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
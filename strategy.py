#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian(20) upper band AND 1d EMA50 is rising AND volume > 1.5 * 20-period avg volume
# Short when price breaks below 4h Donchian(20) lower band AND 1d EMA50 is falling AND volume > 1.5 * 20-period avg volume
# Exit when price crosses the 4h Donchian(20) midpoint
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Donchian(20) provides clear structure, 1d EMA50 filters counter-trend noise, volume confirms conviction
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations)

name = "4h_Donchian20_Breakout_1dEMA50_Trend_Volume_Session"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
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
    
    # Align 4h Donchian levels to 4h timeframe (no shift needed as we're on same TF)
    upper_4h_aligned = upper_4h
    lower_4h_aligned = lower_4h
    mid_4h_aligned = mid_4h
    
    # Get 1d data ONCE before loop for EMA50 trend filter
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
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(mid_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band, EMA50 rising, volume spike
            if (close[i] > upper_4h_aligned[i] and close[i-1] <= upper_4h_aligned[i-1] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower band, EMA50 falling, volume spike
            elif (close[i] < lower_4h_aligned[i] and close[i-1] >= lower_4h_aligned[i-1] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h Donchian midpoint
            if close[i] < mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 4h Donchian midpoint
            if close[i] > mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
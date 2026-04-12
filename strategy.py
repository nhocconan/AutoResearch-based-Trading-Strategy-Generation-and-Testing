#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    highest_20 = np.zeros(len(df_1d))
    lowest_20 = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 19:
            highest_20[i] = np.nan
            lowest_20[i] = np.nan
        else:
            highest_20[i] = np.max(high_1d[i-19:i+1])
            lowest_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 50-period EMA on weekly data
    ema_50 = np.zeros(len(df_1w))
    ema_50[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(df_1w)):
        ema_50[i] = alpha * close_1w[i] + (1 - alpha) * ema_50[i-1]
    
    # Align weekly EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume filter - 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above 20-day high with volume confirmation and weekly uptrend
        long_signal = close[i] > highest_20_aligned[i] and volume_ok[i] and close[i] > ema_50_aligned[i]
        # Short: price breaks below 20-day low with volume confirmation and weekly downtrend
        short_signal = close[i] < lowest_20_aligned[i] and volume_ok[i] and close[i] < ema_50_aligned[i]
        
        # Exit when price returns to the midpoint of the Donchian channel
        midpoint = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2.0
        exit_long = close[i] < midpoint
        exit_short = close[i] > midpoint
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
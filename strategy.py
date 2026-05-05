#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA50) and 1d volume confirmation
# Long when price breaks above 6h Donchian upper band AND close > 12h EMA50 AND 1d volume > 1.5x 20-period average
# Short when price breaks below 6h Donchian lower band AND close < 12h EMA50 AND 1d volume > 1.5x 20-period average
# Exit when price crosses 6h Donchian middle band (mean reversion to median)
# Uses 6h primary timeframe with 12h HTF for trend filter and 1d HTF for volume confirmation
# Volume confirmation ensures breakouts have conviction; Donchian middle band exit reduces whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe

name = "6h_Donchian20_Breakout_12hEMA50_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume filter to 6h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian(20) channels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: median of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 6h timeframe (same df_6h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND above 12h EMA50 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND below 12h EMA50 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian channel breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian(20) upper band AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 4h Donchian(20) lower band AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back below/above 4h Donchian(20) middle band OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides robust price channels, EMA34 filters higher timeframe trend
# Volume confirmation reduces false breakouts, works in both bull and bear markets via trend filter

name = "4h_Donchian20_1dEMA34_VolumeConfirm"
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
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least one completed 4h bar for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_4h = high_20
    lower_4h = low_20
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (wait for completed 4h bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper band, above 1d EMA34, volume confirmation, in session
            if close[i] > upper_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian lower band, below 1d EMA34, volume confirmation, in session
            elif close[i] < lower_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian middle band OR volume drops below average
            if close[i] < middle_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian middle band OR volume drops below average
            if close[i] > middle_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
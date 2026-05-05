#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND price > 1d EMA50 (uptrend) AND volume > 1.8x 20-period average
# Short when price breaks below 4h Donchian lower(20) AND price < 1d EMA50 (downtrend) AND volume > 1.8x 20-period average
# Exit when price crosses 4h Donchian midpoint OR EMA50 filter reverses
# Uses Donchian channels for clear trend structure, 1d EMA50 for regime filter (avoid whipsaws in chop)
# Volume spike confirms institutional participation
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Mid = (Upper + Lower) / 2
    high_rolling_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_20 = high_rolling_max
    lower_20 = low_rolling_min
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation on 4h (threshold: 1.8x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper(20) AND price > EMA50 (uptrend) AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower(20) AND price < EMA50 (downtrend) AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint OR price < EMA50 (trend weakening)
            if close[i] < midpoint_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint OR price > EMA50 (trend weakening)
            if close[i] > midpoint_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
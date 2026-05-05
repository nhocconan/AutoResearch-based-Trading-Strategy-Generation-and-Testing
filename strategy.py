#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume spike
# Long when price breaks above 4h Donchian upper AND 12h EMA50 > 12h EMA200 (uptrend) AND volume spike
# Short when price breaks below 4h Donchian lower AND 12h EMA50 < 12h EMA200 (downtrend) AND volume spike
# Exit when price crosses the 4h Donchian middle (mean) OR trend flips (EMA50 crosses EMA200)
# Uses Donchian channels for structure, 12h dual-EMA for trend filtering (avoid whipsaws in ranging markets)
# Volume spike confirms institutional participation at breakouts
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Donchian20_12hEMA50_200_Trend_VolumeSpike"
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
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels
    high_ma_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 12h data ONCE before loop for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_50_above_200 = ema_50 > ema_200  # True for uptrend, False for downtrend
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    ema_50_above_200_aligned = align_htf_to_ltf(prices, df_12h, ema_50_above_200.astype(float))
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_above_200_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h EMA50 > EMA200 (uptrend) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                ema_50_above_200_aligned[i] > 0.5 and  # Treat as boolean
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h EMA50 < EMA200 (downtrend) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_50_above_200_aligned[i] < 0.5 and  # Treat as boolean
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR trend flips to downtrend
            if close[i] < donchian_middle_aligned[i] or ema_50_above_200_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR trend flips to uptrend
            if close[i] > donchian_middle_aligned[i] or ema_50_above_200_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
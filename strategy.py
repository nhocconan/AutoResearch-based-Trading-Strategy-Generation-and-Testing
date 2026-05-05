#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 6h Donchian upper (20) AND weekly pivot > weekly EMA34 (bullish bias) AND volume spike
# Short when price breaks below 6h Donchian lower (20) AND weekly pivot < weekly EMA34 (bearish bias) AND volume spike
# Weekly pivot provides higher-timeframe structure to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 6h

name = "6h_Donchian20_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for pivot and EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly pivot from previous completed weekly bar (HLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].values
    
    # Shift by 1 to use only completed weekly bar (look-ahead safety)
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    close_1w_shifted = np.roll(close_1w, 1)
    
    # Weekly pivot point (PP) = (H+L+C)/3
    weekly_pivot = (high_1w_shifted + low_1w_shifted + close_1w_shifted) / 3.0
    # Weekly bias: 1 if pivot > EMA34 (bullish), -1 if pivot < EMA34 (bearish)
    weekly_bias = np.where(weekly_pivot > ema_34_1w, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = high_roll_max
        donchian_lower = low_roll_min
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly bullish bias AND volume spike
            if (close[i] > donchian_upper[i] and 
                weekly_bias_aligned[i] == 1 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly bearish bias AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  weekly_bias_aligned[i] == -1 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian lower OR weekly bias turns bearish
            if close[i] < donchian_lower[i] or weekly_bias_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian upper OR weekly bias turns bullish
            if close[i] > donchian_upper[i] or weekly_bias_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
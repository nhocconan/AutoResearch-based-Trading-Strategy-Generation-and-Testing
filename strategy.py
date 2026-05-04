#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike (>2.0x 50 EMA volume)
# Uses Donchian levels from prior completed 1d bar for structure
# 1d EMA50 ensures we only trade in direction of higher timeframe trend
# Volume confirmation ensures breakout has institutional participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# This avoids overtrading by using strong breakout levels and strict trend filter
# Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Shift EMA by 1 to use only prior completed 1d bar (no look-ahead)
    ema_50_shifted = np.roll(ema_50, 1)
    ema_50_shifted[0] = np.nan
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_shifted)
    
    # Volume confirmation: 50-period EMA of volume on 12h timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian levels (20-period) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 1d bar
    upper_shifted = np.roll(upper, 1)
    lower_shifted = np.roll(lower, 1)
    upper_shifted[0] = np.nan
    lower_shifted[0] = np.nan
    
    # Align Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_shifted)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_50[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + price > EMA50 (uptrend) + volume spike
            if close[i] > upper_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + price < EMA50 (downtrend) + volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian levels OR price crosses below EMA50 (trend change)
            midpoint = (upper_aligned[i] + lower_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian levels OR price crosses above EMA50 (trend change)
            midpoint = (upper_aligned[i] + lower_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
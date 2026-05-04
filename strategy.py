#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts from 4h for structure, aligned with 12h EMA50 trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 minimizes fee churn while targeting 75-200 total trades over 4 years
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_12hEMA50_VolumeConfirm_Balanced"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Calculate Donchian(20) channels from 4h data
    # Upper channel: highest high over 20 periods
    # Lower channel: lowest low over 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND price > 12h EMA50 AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND price < 12h EMA50 AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower OR price crosses below 12h EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper OR price crosses above 12h EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
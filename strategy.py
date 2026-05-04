#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Donchian channel from prior completed 4h bar for structure (breakout = momentum)
# 12h EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
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
    if len(df_12h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Calculate Donchian(20) from prior completed 4h bar (using mtf_data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channel: upper = max(high,20), lower = min(low,20)
    high_rolling_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    upper_shifted = np.roll(high_rolling_max, 1)
    lower_shifted = np.roll(low_rolling_min, 1)
    upper_shifted[0] = np.nan
    lower_shifted[0] = np.nan
    
    # Align Donchian levels to 4h timeframe (already aligned, just need shift)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_shifted)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band AND price > 12h EMA50 AND volume spike
            if close[i] > upper_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian band AND price < 12h EMA50 AND volume spike
            elif close[i] < lower_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian band OR price crosses below 12h EMA50
            if close[i] < lower_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian band OR price crosses above 12h EMA50
            if close[i] > upper_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
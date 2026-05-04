#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.6x 20 EMA volume)
# Uses 4h Donchian channel (20-period high/low) for structure - proven breakout strategy
# 1d EMA50 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation requires significant participation (>1.6x average volume) to filter false breakouts
# Discrete sizing 0.28 balances risk and return while minimizing fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 4h timeframe
# Works in both bull (continuation at upper channel) and bear (continuation at lower channel) markets
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channel (20-period) - using only completed bars
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND price > 1d EMA50 AND volume spike
            if close[i] > highest_20[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = 0.28
                position = 1
            # Short conditions: price breaks below lower Donchian AND price < 1d EMA50 AND volume spike
            elif close[i] < lowest_20[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian OR price crosses below 1d EMA50
            if close[i] < lowest_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price returns to upper Donchian OR price crosses above 1d EMA50
            if close[i] > highest_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
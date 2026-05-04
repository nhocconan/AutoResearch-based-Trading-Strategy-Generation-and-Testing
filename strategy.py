#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses 6h Donchian channel (20-bar high/low) for structural breakouts - captures strong momentum moves
# 1w EMA200 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear
# Volume confirmation filters false breakouts (>1.5x average volume)
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "6h_Donchian20_1wEMA200_VolumeConfirm"
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA200 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(200) trend filter from prior completed 1w bar
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_shifted = np.roll(ema_200_1w, 1)
    ema_200_1w_shifted[0] = np.nan
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w_shifted)
    
    # Calculate 6h Donchian channel (20-period) from current completed 6h bar
    # We need to calculate this on the LTF but ensure no look-ahead
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high AND price > 1w EMA200 AND volume spike
            vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
            if close[i] > high_max[i] and close[i] > ema_200_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20.iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-period low AND price < 1w EMA200 AND volume spike
            elif close[i] < low_min[i] and close[i] < ema_200_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20.iloc[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 20-period low OR price crosses below 1w EMA200
            if close[i] < low_min[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 20-period high OR price crosses above 1w EMA200
            if close[i] > high_max[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
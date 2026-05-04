#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses Donchian channel from prior completed 6h bar for structure (breakout = momentum)
# Weekly EMA200 filter ensures we trade only in direction of major trend (avoids counter-trend whipsaws in ranging markets)
# Volume confirmation requires >2.0x average volume to ensure breakout has institutional participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (breakout continuation) and bear markets (breakdown continuation when below weekly EMA200)
# Focus on BTC/ETH by requiring weekly trend alignment (avoids SOL-only bias, more robust across regimes)

name = "6h_Donchian20_WeeklyEMA200_VolumeConfirm"
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
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(200) trend filter from prior completed weekly bar
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_shifted = np.roll(ema_200_1w, 1)
    ema_200_1w_shifted[0] = np.nan
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w_shifted)
    
    # Get 6h data for Donchian channel and volume EMA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channel: upper = max(high,20), lower = min(low,20)
    high_rolling_max = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 6h bar (no look-ahead)
    upper_shifted = np.roll(high_rolling_max, 1)
    lower_shifted = np.roll(low_rolling_min, 1)
    upper_shifted[0] = np.nan
    lower_shifted[0] = np.nan
    
    # Align Donchian levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_shifted)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band AND price > weekly EMA200 AND volume spike (>2.0x)
            if close[i] > upper_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian band AND price < weekly EMA200 AND volume spike (>2.0x)
            elif close[i] < lower_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian band OR price crosses below weekly EMA200
            if close[i] < lower_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian band OR price crosses above weekly EMA200
            if close[i] > upper_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
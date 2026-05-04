#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses Donchian channel from prior completed 4h bar for structure (breakout at upper/lower band = momentum)
# 1d EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.30 balances risk and return while minimizing fee churn
# Target: 100-200 total trades over 4 years = 25-50/year for 4h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias, more robust across regimes)

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
    if len(df_1d) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) channel from prior completed 4h bar
    # Upper band = 20-period high, Lower band = 20-period low
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    upper_20_shifted = np.roll(upper_20, 1)
    lower_20_shifted = np.roll(lower_20, 1)
    upper_20_shifted[0] = np.nan
    lower_20_shifted[0] = np.nan
    
    # Align Donchian bands to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20_shifted)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band AND price > 1d EMA50 AND volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below lower Donchian band AND price < 1d EMA50 AND volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian band OR price crosses below 1d EMA50
            if close[i] < lower_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to upper Donchian band OR price crosses above 1d EMA50
            if close[i] > upper_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
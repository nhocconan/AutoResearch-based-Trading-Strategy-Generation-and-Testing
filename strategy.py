#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Uses Donchian channel from weekly timeframe for major trend structure
# 1w EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts
# Discrete sizing 0.25 minimizes fee churn while targeting 30-100 total trades over 4 years
# Works in bull markets (continuation at upper band) and bear markets (continuation at lower band)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_Donchian20_1wEMA34_VolumeConfirm_Balanced"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian calculation and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate prior completed 1w Donchian levels (20-period)
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high_1w).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=lookback, min_periods=lookback).min().values
    
    # Shift to use prior completed 1w bar (avoid look-ahead)
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    
    # Align Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_shifted)
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian Upper AND price > 1w EMA34 AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian Lower AND price < 1w EMA34 AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian Lower OR price crosses below 1w EMA34
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian Upper OR price crosses above 1w EMA34
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
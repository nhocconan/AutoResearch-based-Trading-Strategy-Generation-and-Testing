#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channel from prior 20 periods (4h) for structure, 1d EMA34 for trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear: EMA34 ensures we only trade with the trend, Donchian provides objective breakout levels.

name = "4h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian channel (20-period) on 4h timeframe - use prior completed period only
    # We need 20 periods of data, so lookback 20 bars excluding current
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + price above 1d EMA34 + volume spike
            if close[i] > highest_20[i] and close[i] > ema_34_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + price below 1d EMA34 + volume spike
            elif close[i] < lowest_20[i] and close[i] < ema_34_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1d EMA34
            donchian_mid = (highest_20[i] + lowest_20[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1d EMA34
            donchian_mid = (highest_20[i] + lowest_20[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
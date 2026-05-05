#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume spike
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume spike
# Exit when price crosses Donchian(10) midpoint OR trend filter fails
# Uses Donchian for structure, EMA34 for trend filter, volume for conviction
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels on 12h
    if len(high) >= 20:
        # Donchian(20) high/low
        donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian(10) midpoint for exit
        donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    else:
        donchian_high_20 = np.full(n, np.nan)
        donchian_low_20 = np.full(n, np.nan)
        donchian_mid_10 = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian(20) high AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > donchian_high_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian(20) low AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < donchian_low_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian(10) midpoint OR price < 1d EMA34 (trend fail)
            if close[i] < donchian_mid_10[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian(10) midpoint OR price > 1d EMA34 (trend fail)
            if close[i] > donchian_mid_10[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
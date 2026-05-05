#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike
# Long when price breaks above upper Donchian(20) AND close > 1d EMA34 (uptrend) AND volume spike
# Short when price breaks below lower Donchian(20) AND close < 1d EMA34 (downtrend) AND volume spike
# Uses Donchian channels for structure, EMA34 for trend filter (avoid counter-trend), volume spike for conviction.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe to avoid fee drag.
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).

name = "12h_Donchian20_1dEMA34_VolumeSpike"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h timeframe
    if len(high) >= 20:
        high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        high_ma_20 = np.full(n, np.nan)
        low_ma_20 = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend (price > EMA34) AND volume spike
            if (close[i] > high_ma_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend (price < EMA34) AND volume spike
            elif (close[i] < low_ma_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper Donchian OR closes below EMA34
            if close[i] < high_ma_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower Donchian OR closes above EMA34
            if close[i] > low_ma_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
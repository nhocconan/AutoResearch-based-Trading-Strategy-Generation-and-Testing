#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation
# This strategy trades breakouts of the 4h Donchian channel (20-period) with trend alignment
# from higher timeframe (12h) EMA and volume confirmation. It works in both bull and bear
# markets by following the trend direction on higher timeframe. Uses discrete position sizing
# (0.30) to balance return and minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + above 12h EMA + volume spike
            if close[i] > high_max_20[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below lower Donchian + below 12h EMA + volume spike
            elif close[i] < low_min_20[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses 12h EMA in opposite direction
            if position == 1:
                # Exit long: Close below 12h EMA
                if close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Exit short: Close above 12h EMA
                if close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0
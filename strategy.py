#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrend_v1
Hypothesis: Combines Donchian channel breakout (20-period) with volume confirmation and 12h EMA trend filter.
Captures breakouts in both bull and bear markets while filtering false signals with volume and trend.
Target: 20-30 trades/year to minimize fee drag. Works in bull via breakouts, bear via short breakdowns.
"""

name = "4h_DonchianBreakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h trend filter: EMA of 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i]) or 
            vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian with volume and trend confirmation
            if close[i] > high_roll[i] and volume[i] > vol_ma[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian with volume and trend confirmation
            elif close[i] < low_roll[i] and volume[i] > vol_ma[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below lower Donchian
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above upper Donchian
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
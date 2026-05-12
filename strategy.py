#!/usr/bin/env python3
"""
4h_MultiTimeframe_Volume_Trend_Filter
Hypothesis: Combines 4h price action with 1d volume confirmation and 1w trend filter to capture high-probability breakouts.
Uses 4h Donchian breakout (20-period) with 1d volume spike (>2x 20-period average) and 1w EMA trend filter.
Only takes long in 1d uptrend, short in 1d downtrend to avoid counter-trend trades.
Position size fixed at 0.25 to balance risk and return. Designed for 20-40 trades/year per symbol.
Works in bull/bear by following 1d trend direction and requiring volume confirmation.
"""

name = "4h_MultiTimeframe_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume spike (>2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1w EMA trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian upper + volume spike + 1w uptrend
            if (close[i] > high_roll[i] and 
                volume_spike[i] and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 4h Donchian lower + volume spike + 1w downtrend
            elif (close[i] < low_roll[i] and 
                  volume_spike[i] and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h Donchian lower
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 4h Donchian upper
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
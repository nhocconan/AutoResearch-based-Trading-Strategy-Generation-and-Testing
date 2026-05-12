#!/usr/bin/env python3
# 12h_Donchian_20_Breakout_1wTrend_Volume
# Hypothesis: Donchian channel breakout on 12h with 1w trend filter and volume confirmation.
# The 1w trend filter ensures we only trade in the direction of the weekly trend,
# avoiding counter-trend trades in ranging markets. Volume confirmation ensures
# breakouts have institutional backing. This combination reduces false breakouts
# and works in both bull and bear markets by following the higher timeframe trend.
# Designed for low trade frequency to minimize fee drag on 12h timeframe.

name = "12h_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # === 1w EMA34 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h Donchian Channel (20) ===
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_ma_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_ma_20[i-1]  # Break below previous period's low
        
        # 1w trend filter: price relative to EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout up, uptrend, volume confirmation
            if breakout_up and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout down, downtrend, volume confirmation
            elif breakout_down and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakout down or trend change to downtrend
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout up or trend change to uptrend
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
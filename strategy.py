#!/usr/bin/env python3
# 4h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Donchian channel breakout on 4h with 1d trend filter and volume confirmation.
# Breakouts above the 20-period high or below the 20-period low signal momentum shifts.
# The 1d EMA50 provides trend context to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.
# Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "4h_Donchian_Breakout_1dTrend_Volume"
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout above Donchian high, uptrend, volume confirmation
            if close[i] > donchian_high[i] and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below Donchian low, downtrend, volume confirmation
            elif close[i] < donchian_low[i] and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below Donchian low or trend reversal
            if close[i] < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above Donchian high or trend reversal
            if close[i] > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
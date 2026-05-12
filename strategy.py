#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Donchian channel breakout on 12h with 1d EMA trend filter and volume confirmation.
# Donchian channels provide clear breakout signals based on price extremes.
# The 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by following the trend defined by higher timeframe.

name = "12h_Donchian_Breakout_1dTrend_Volume"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Donchian Channel (20) on 12h ===
    # Upper band: highest high over last 20 periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band with volume and higher timeframe uptrend
            if (close[i] > high_20[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band with volume and higher timeframe downtrend
            elif (close[i] < low_20[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below lower Donchian band or higher timeframe trend changes
            if (close[i] < low_20[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper Donchian band or higher timeframe trend changes
            if (close[i] > high_20[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
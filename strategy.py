#!/usr/bin/env python3
# 12h_Donchian_Breakout_1wTrend_Volume
# Hypothesis: Donchian channel breakout on 12h with 1-week EMA trend filter and volume confirmation.
# The 1-week EMA ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by following the trend defined by higher timeframe.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Donchian_Breakout_1wTrend_Volume"
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
    
    # === 1-week Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 30-period EMA on 1w for trend direction
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # === Donchian Channel (20) on 12h ===
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_30_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1w EMA
        trend_up = close[i] > ema_30_1w_aligned[i]
        trend_down = close[i] < ema_30_1w_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band with volume and higher timeframe uptrend
            if (close[i] > upper_band[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band with volume and higher timeframe downtrend
            elif (close[i] < lower_band[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below lower band or higher timeframe trend changes
            if (close[i] < lower_band[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper band or higher timeframe trend changes
            if (close[i] > upper_band[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
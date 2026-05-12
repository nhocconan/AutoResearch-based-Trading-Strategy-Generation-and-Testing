#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Camarilla pivot breakout on 1h with 4h EMA trend filter and volume confirmation.
# Camarilla pivots identify key support/resistance levels. The 4h EMA ensures alignment with higher timeframe trend.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by following the trend defined by higher timeframe.
# Target: 15-37 trades/year on 1h timeframe.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 34-period EMA on 4h for trend direction
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Camarilla Pivot Levels (based on previous day) ===
    # We'll use daily OHLC from 1d timeframe for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1 based on previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0 / 12)
    s1 = pivot - (range_1d * 1.0 / 12)
    
    # Align daily levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation (24-period average on 1h) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and higher timeframe uptrend
            if (close[i] > r1_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume and higher timeframe downtrend
            elif (close[i] < s1_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below S1 or higher timeframe trend changes
            if (close[i] < s1_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 or higher timeframe trend changes
            if (close[i] > r1_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
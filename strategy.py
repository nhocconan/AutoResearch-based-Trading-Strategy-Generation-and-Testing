#!/usr/bin/env python3
# 1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use daily Camarilla pivot levels (R1, S1) with weekly trend filter and volume confirmation.
# Long when price breaks above R1 with volume, short when breaks below S1 with volume, only in direction of weekly trend.
# Designed for low frequency (10-25 trades/year) to capture momentum with tight entries, reducing fee drag.
# Weekly trend: price above/below weekly EMA34. Works in both bull and bear markets by aligning with higher timeframe trend.

name = "1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # === Weekly EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla pivot levels (R1, S1) ===
    # Calculate from previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla formulas
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, in uptrend
            if close[i] > R1[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume, in downtrend
            elif close[i] < S1[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below S1 or trend reversal
            if close[i] < S1[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or trend reversal
            if close[i] > R1[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
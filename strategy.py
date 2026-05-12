#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter
# Hypothesis: Use Camarilla pivot levels from daily timeframe for breakout entries with trend filter and volume confirmation.
# Long when price breaks above R1 with price above daily EMA34 and volume > 2x 20-period volume MA.
# Short when price breaks below S1 with price below daily EMA34 and volume > 2x 20-period volume MA.
# Exit when price reverts to the daily pivot point (PP).
# Designed to capture strong intraday moves with multi-timeframe confirmation, works in both bull and bear markets by filtering with daily trend.
# Targets 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter"
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
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Daily EMA for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(daily_ema_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with price > daily EMA and volume > 2x MA
            if close[i] > R1_aligned[i] and close[i] > daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with price < daily EMA and volume > 2x MA
            elif close[i] < S1_aligned[i] and close[i] < daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to daily pivot point
            if close[i] <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to daily pivot point
            if close[i] >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend
# Hypothesis: Use Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h timeframe
# with 1d EMA34 trend filter and volume confirmation. Long when Bull Power > 0 and rising, Bear Power < 0,
# and price above 1d EMA34. Short when Bear Power < 0 and falling, Bull Power > 0, and price below 1d EMA34.
# Designed for low frequency (15-30 trades/year) to capture momentum in both bull and bear markets
# by aligning with higher timeframe trend while using Elder Ray for precise entry/exit.

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
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
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Elder Ray Index on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray conditions
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Bull Power positive and rising, Bear Power negative, uptrend, volume
            if bull_positive and bull_rising and bear_negative and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative and falling, Bull Power positive, downtrend, volume
            elif bear_negative and bear_falling and bull_positive and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bear Power becomes positive or trend reversal
            if bear_power[i] > 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes negative or trend reversal
            if bull_power[i] < 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
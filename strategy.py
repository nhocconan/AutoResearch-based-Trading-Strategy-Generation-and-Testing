#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and 12h volume confirmation.
# Long when Alligator jaws (SMMA13) > teeth (SMMA8) > lips (SMMA5) and price > 1d EMA50 and 12h volume > 1.5x 20-period average.
# Short when Alligator jaws < teeth < lips and price < 1d EMA50 and 12h volume > 1.5x 20-period average.
# Exit when Alligator condition reverses (jaws < teeth or jaws > lips depending on direction).
# Uses Williams Alligator for trend identification, 1d EMA50 for higher timeframe trend filter, and volume to confirm participation.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsAlligator_1dEMA50_12hVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # 12h Williams Alligator: SMMA(13), SMMA(8), SMMA(5) on median price
    median_price = (high + low) / 2.0
    jaws = smma(median_price, 13)   # Alligator's Jaw (blue, 13-period)
    teeth = smma(median_price, 8)   # Alligator's Teeth (red, 8-period)
    lips = smma(median_price, 5)    # Alligator's Lips (green, 5-period)
    
    # 12h volume confirmation: > 1.5x 20-period average (balanced filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator aligned (jaws > teeth > lips) + price > 1d EMA50 + volume confirmation
            if (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned (jaws < teeth < lips) + price < 1d EMA50 + volume confirmation
            elif (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator loses alignment (jaws <= teeth or teeth <= lips)
            if jaws[i] <= teeth[i] or teeth[i] <= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator loses alignment (jaws >= teeth or teeth >= lips)
            if jaws[i] >= teeth[i] or teeth[i] >= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
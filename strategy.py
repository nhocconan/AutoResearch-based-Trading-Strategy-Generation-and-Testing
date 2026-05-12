#!/usr/bin/env python3
# 4h Williams Alligator with Volume Spike and 1d Trend Filter
# Hypothesis: Williams Alligator identifies trends via smoothed moving averages (Jaws, Teeth, Lips).
# When Lips > Teeth > Jaws = uptrend; Lips < Teeth < Jaws = downtrend.
# Combines with volume spike for momentum confirmation and daily EMA50 trend filter.
# Works in bull and bear markets by following Alligator-defined trends.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_Williams_Alligator_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Alligator (13, 8, 5 periods) ===
    # Jaws: SMMA(13) of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaws_raw = smma(median_price, 13)
    jaws = np.roll(jaws_raw, 8)  # Shift 8 bars forward
    
    # Teeth: SMMA(8) of median price, shifted 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift 5 bars forward
    
    # Lips: SMMA(5) of median price
    lips = smma(median_price, 5)
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaws (uptrend) + volume spike + price above daily EMA50
            if (lips[i] > teeth[i] and teeth[i] > jaws[i] and 
                vol_spike[i] and
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (downtrend) + volume spike + price below daily EMA50
            elif (lips[i] < teeth[i] and teeth[i] < jaws[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend weakens (Lips crosses below Teeth)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (Lips crosses above Teeth)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
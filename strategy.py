#!/usr/bin/env python3
# 12h_Trix_1dTrend_Volume
# Hypothesis: Use TRIX (Triple Exponential Average) on 12h for momentum direction,
# filtered by 1d EMA34 trend and volume confirmation. Enter long when TRIX crosses above zero
# and price > 1d EMA34 with volume spike; short when TRIX crosses below zero and price < 1d EMA34
# with volume spike. Exit on opposite TRIX crossover or trend failure. Designed for low frequency
# (10-30 trades/year) to avoid fee drag. TRIX filters noise and works in both bull and bear markets
# by capturing sustained momentum shifts when aligned with higher timeframe trend.

name = "12h_Trix_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def trix(close, period=15):
    """
    Calculate TRIX (Triple Exponential Average).
    Returns TRIX values.
    """
    # First EMA
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Second EMA of first EMA
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Third EMA of second EMA
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate TRIX: (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(close)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # First value is zero (no previous)
    trix_raw[0] = 0
    
    return trix_raw

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 12h data
    trix_val = trix(close, period=15)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable (15*2+10 for TRIX + EMA34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix_val[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # TRIX signals: zero cross
        trix_cross_up = trix_val[i] > 0 and trix_val[i-1] <= 0
        trix_cross_down = trix_val[i] < 0 and trix_val[i-1] >= 0
        
        if position == 0:
            # LONG: TRIX crosses above zero, price above daily EMA34, volume confirmation
            if trix_cross_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, price below daily EMA34, volume confirmation
            elif trix_cross_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend fails
            if trix_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend fails
            if trix_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
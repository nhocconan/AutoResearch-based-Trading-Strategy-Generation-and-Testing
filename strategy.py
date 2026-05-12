#!/usr/bin/env python3
# 12h_1d_Camarilla_R3S3_Breakout_Volume
# Hypothesis: Breakout beyond Camarilla R3/S3 levels on 12h with 1d trend filter and volume confirmation.
# The Camarilla R3/S3 levels represent strong intraday support/resistance.
# Using 1d trend (price vs EMA34) ensures we trade with higher timeframe momentum.
# Volume confirmation adds conviction to breakouts. Designed for low frequency to avoid fee drag.

name = "12h_1d_Camarilla_R3S3_Breakout_Volume"
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
    
    # === 1d Camarilla Levels (based on previous day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values  # Previous day high
    low_prev = df_1d['low'].shift(1).values    # Previous day low
    close_prev = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla R3 and S3 levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 2
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # === 1d Trend Filter (EMA34) ===
    ema_34_1d = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Confirmation (20-period average on 12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Price relative to Camarilla levels
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R3, uptrend, volume confirmation
            if price_above_R3 and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3, downtrend, volume confirmation
            elif price_below_S3 and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or trend changes
            if not price_above_R3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or trend changes
            if not price_below_S3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
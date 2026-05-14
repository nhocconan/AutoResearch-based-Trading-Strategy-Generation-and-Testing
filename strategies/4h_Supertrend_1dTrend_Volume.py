#!/usr/bin/env python3
# 4h_Supertrend_1dTrend_Volume
# Hypothesis: Supertrend on 4h with 1d EMA trend filter and volume confirmation.
# Supertrend adapts to volatility and captures trends effectively. The 1d EMA filter ensures
# alignment with higher timeframe trend, reducing counter-trend trades. Volume confirmation
# ensures breakouts have conviction. Designed to work in both bull and bear markets by
# following the trend defined by higher timeframe.

name = "4h_Supertrend_1dTrend_Volume"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 30-period EMA on 1d for trend direction
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # === Supertrend (10, 3.0) on 4h ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + 3.0 * atr_10
    lower_band = hl_avg - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    final_upper = np.copy(upper_band)
    final_lower = np.copy(lower_band)
    
    for i in range(1, len(final_upper)):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
            
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # Supertrend
    supertrend = np.zeros(len(close))
    for i in range(len(supertrend)):
        if i == 0:
            supertrend[i] = 0  # undefined
        elif close[i] > final_upper[i-1]:
            supertrend[i] = 1  # uptrend
        elif close[i] < final_lower[i-1]:
            supertrend[i] = -1  # downtrend
        else:
            supertrend[i] = supertrend[i-1]  # continue previous trend
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_30_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_30_1d_aligned[i]
        trend_down = close[i] < ema_30_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Supertrend uptrend with volume and higher timeframe uptrend
            if (supertrend[i] == 1 and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend with volume and higher timeframe downtrend
            elif (supertrend[i] == -1 and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Supertrend turns down or higher timeframe trend changes
            if (supertrend[i] == -1 or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend turns up or higher timeframe trend changes
            if (supertrend[i] == 1 or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
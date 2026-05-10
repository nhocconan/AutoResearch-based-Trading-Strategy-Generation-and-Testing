#!/usr/bin/env python3
# 6h_VolumeWeighted_VWAP_Rebound_1wTrend
# Hypothesis: Combines VWAP mean reversion on 6h with weekly trend filter. Price tends to revert to VWAP
# during ranging markets, but in strong weekly trends, VWAP acts as dynamic support/resistance.
# Long when: price crosses above VWAP from below AND weekly uptrend AND volume above average.
# Short when: price crosses below VWAP from above AND weekly downtrend AND volume above average.
# Uses volume confirmation to avoid false breakouts. Works in both bull and bear markets by
# aligning with higher timeframe trend while capturing mean reversion entries.
# Expected low trade frequency due to multiple confluence requirements.

name = "6h_VolumeWeighted_VWAP_Rebound_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP on 6h
    vwap_6h = vwap(high, low, close, volume)
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (50) and VWAP/volume calculations
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vwap_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from weekly EMA
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_ma[i]
        
        if position == 0:
            # Long: price crosses above VWAP from below, weekly uptrend, volume confirmation
            if (close[i] > vwap_6h[i] and close[i-1] <= vwap_6h[i-1] and 
                uptrend and vol_above_avg):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP from above, weekly downtrend, volume confirmation
            elif (close[i] < vwap_6h[i] and close[i-1] >= vwap_6h[i-1] and 
                  downtrend and vol_above_avg):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP
            if close[i] < vwap_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP
            if close[i] > vwap_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
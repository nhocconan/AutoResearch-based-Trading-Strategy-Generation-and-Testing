#!/usr/bin/env python3
"""
4h_SuperTrend_EMA200_Turn_Confirmation
Hypothesis: SuperTrend(10,3) with EMA200 as trend filter and momentum confirmation (close > EMA50) works in both bull and bear markets.
Enter long when SuperTrend flips to uptrend AND price > EMA200 AND close > EMA50.
Enter short when SuperTrend flips to downtrend AND price < EMA200 AND close < EMA50.
Exit on SuperTrend reversal. Uses 1d SuperTrend as higher timeframe filter to avoid counter-trend trades.
Target: 25-50 trades/year per symbol.
"""

name = "4h_SuperTrend_EMA200_Turn_Confirmation"
timeframe = "4h"
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
    
    # SuperTrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    
    for i in range(1, n):
        final_upper[i] = upper_band[i] if (upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lower_band[i] if (lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
    
    # SuperTrend
    super_trend = np.zeros(n)
    super_trend[0] = final_lower[0]
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        super_trend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
    
    # EMA200 for long-term trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # EMA50 for momentum confirmation
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d SuperTrend as HTF filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SuperTrend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[0.0], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2_1d = (high_1d + low_1d) / 2
    upper_band_1d = hl2_1d + multiplier * atr_1d
    lower_band_1d = hl2_1d - multiplier * atr_1d
    
    final_upper_1d = np.zeros(len(df_1d))
    final_lower_1d = np.zeros(len(df_1d))
    final_upper_1d[0] = upper_band_1d[0]
    final_lower_1d[0] = lower_band_1d[0]
    
    for i in range(1, len(df_1d)):
        final_upper_1d[i] = upper_band_1d[i] if (upper_band_1d[i] < final_upper_1d[i-1] or close_1d[i-1] > final_upper_1d[i-1]) else final_upper_1d[i-1]
        final_lower_1d[i] = lower_band_1d[i] if (lower_band_1d[i] > final_lower_1d[i-1] or close_1d[i-1] < final_lower_1d[i-1]) else final_lower_1d[i-1]
    
    super_trend_1d = np.zeros(len(df_1d))
    direction_1d = np.ones(len(df_1d))
    super_trend_1d[0] = final_lower_1d[0]
    direction_1d[0] = 1
    
    for i in range(1, len(df_1d)):
        if close_1d[i] > final_upper_1d[i-1]:
            direction_1d[i] = 1
        elif close_1d[i] < final_lower_1d[i-1]:
            direction_1d[i] = -1
        else:
            direction_1d[i] = direction_1d[i-1]
        
        super_trend_1d[i] = final_lower_1d[i] if direction_1d[i] == 1 else final_upper_1d[i]
    
    # Align 1d SuperTrend direction to 4h
    uptrend_1d = direction_1d == 1
    downtrend_1d = direction_1d == -1
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Wait for EMA200
        # Get values
        st_dir = direction[i]  # 1 for uptrend, -1 for downtrend
        price = close[i]
        ema200 = ema_200[i]
        ema50 = ema_50[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: SuperTrend uptrend, price > EMA200, close > EMA50, 1d uptrend filter
            if st_dir == 1 and price > ema200 and close[i] > ema50 and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: SuperTrend downtrend, price < EMA200, close < EMA50, 1d downtrend filter
            elif st_dir == -1 and price < ema200 and close[i] < ema50 and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: SuperTrend turns down
            if st_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: SuperTrend turns up
            if st_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_TrendFollow_V1
# Hypothesis: On 6h timeframe, follow trends confirmed by weekly pivot levels (weekly pivot as mean).
# In bull markets: price > weekly pivot + weekly RSI > 50 = long.
# In bear markets: price < weekly pivot + weekly RSI < 50 = short.
# Uses weekly RSI to filter regime and avoids counter-trend trades.
# Targets 20-40 trades/year by requiring alignment with weekly structure.
# Works in both bull and bear markets due to adaptive trend filtering.

name = "6h_1d_WeeklyPivot_TrendFollow_V1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot (standard pivot point)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_w)
    avg_loss = np.zeros_like(close_w)
    
    # Wilder smoothing for RSI
    for i in range(len(close_w)):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_w = 100 - (100 / (1 + rs))
    
    # Align weekly data to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(rsi_w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot AND weekly RSI > 50 (bullish bias)
            if (close[i] > pivot_w_aligned[i] * 1.002 and 
                rsi_w_aligned[i] > 50 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND weekly RSI < 50 (bearish bias)
            elif (close[i] < pivot_w_aligned[i] * 0.998 and 
                  rsi_w_aligned[i] < 50 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR RSI turns bearish
            if (close[i] < pivot_w_aligned[i] * 0.998 or 
                rsi_w_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR RSI turns bullish
            if (close[i] > pivot_w_aligned[i] * 1.002 or 
                rsi_w_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
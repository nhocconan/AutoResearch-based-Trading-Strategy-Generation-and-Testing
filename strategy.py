#!/usr/bin/env python3
"""
12h_KAMA_Trend_Rotation_v1
Hypothesis: Uses Kaufman's Adaptive Moving Average (KAMA) on 12h for trend direction,
with 1-day trend filter and volume confirmation to avoid whipsaws. KAMA adapts to market
noise, staying flat in chop and trending in strong moves, reducing false signals.
Targets 15-30 trades/year to minimize fee drag on 12h timeframe.
"""

name = "12h_KAMA_Trend_Rotation_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Kaufman's Adaptive Moving Average (KAMA)
    # Parameters: ER lookback = 10, Fastest EMA = 2, Slowest EMA = 30
    er_lookback = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Calculate ER and Smoothing Constant
    er = np.zeros_like(change)
    sc = np.zeros_like(change)
    
    change_sum = pd.Series(change).rolling(window=er_lookback, min_periods=er_lookback).sum().values
    volatility_sum = pd.Series(volatility).rolling(window=er_lookback, min_periods=er_lookback).sum().values
    
    er = np.divide(change_sum, volatility_sum, out=np.zeros_like(change_sum), where=volatility_sum!=0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_ema + 1.0)
    slow_sc = 2.0 / (slow_ema + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA AND above 1-day EMA with volume confirmation
            if close[i] > kama[i] and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND below 1-day EMA with volume confirmation
            elif close[i] < kama[i] and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA OR below 1-day EMA
            if close[i] < kama[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA OR above 1-day EMA
            if close[i] > kama[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
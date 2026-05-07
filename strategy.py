#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_Volume_Filter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))
    er = np.zeros_like(close)
    er[er_period:] = change[er_period:] / volatility[er_period:]
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA for trend direction (10-period ER)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Weekly trend filter: price above/below weekly KAMA
    weekly_close = df_1w['close'].values
    weekly_kama = calculate_kama(weekly_close, er_period=10, fast_sc=2, slow_sc=30)
    weekly_kama_aligned = align_htf_to_ltf(prices, df_1w, weekly_kama)
    
    # Volume spike filter: > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_kama_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily KAMA AND weekly KAMA with volume spike
            if close[i] > kama[i] and close[i] > weekly_kama_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below daily KAMA AND weekly KAMA with volume spike
            elif close[i] < kama[i] and close[i] < weekly_kama_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below daily KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above daily KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA captures adaptive trend, weekly KAMA filter ensures alignment with higher timeframe trend, volume spike confirms momentum. Works in bull/bear by following adaptive trend with volume confirmation. Target: 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to limit risk and churn.
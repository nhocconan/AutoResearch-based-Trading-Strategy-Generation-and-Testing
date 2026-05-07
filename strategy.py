#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA calculation with ER=10, fast=2, slow=30 on weekly close
    close_1w = df_1w['close'].values
    change_1w = np.abs(close_1w - np.roll(close_1w, 10))
    change_1w[0:10] = 0
    volatility_1w = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        volatility_1w[i] = volatility_1w[i-1] + np.abs(close_1w[i] - close_1w[i-1])
        if i >= 10:
            volatility_1w[i] -= np.abs(close_1w[i-10] - close_1w[i-11]) if i >= 11 else 0
    er_1w = np.where(volatility_1w != 0, change_1w / volatility_1w, 0)
    sc_1w = (er_1w * (2/2 - 2/30) + 2/30) ** 2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly KAMA with volume filter
            if close[i] > kama_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly KAMA with volume filter
            elif close[i] < kama_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below weekly KAMA
            if close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above weekly KAMA
            if close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
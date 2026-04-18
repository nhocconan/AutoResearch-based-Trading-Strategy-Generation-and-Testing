#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Filter_v2
Hypothesis: Use KAMA (14,2,30) for trend direction, volume > 1.5x 20-period average for confirmation, and enter only when price crosses above/below KAMA with volume confirmation. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Volume filter ensures institutional participation. Designed for 15-30 trades/year to avoid fee drag.
"""

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
    
    # KAMA: Efficiency Ratio (ER) smoothing
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        close_series = pd.Series(close)
        change = abs(close_series.diff(er_period))
        volatility = close_series.diff().abs().rolling(window=er_period).sum()
        er = change / volatility.replace(0, np.nan)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close, dtype=float)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc.iloc[i]) if hasattr(sc, 'iloc') else np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation
            if price > kama_val and close[i-1] <= kama[i-1] and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation
            elif price < kama_val and close[i-1] >= kama[i-1] and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA
            if price < kama_val and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA
            if price > kama_val and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0
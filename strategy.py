#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_Volume_Regime_v1
Hypothesis: KAMA adapts to market efficiency - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with 1d volume regime (high/low volume days)
to filter false signals. Works in both bull/bear: KAMA captures trend, volume regime
avoids chop whipsaws. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_With_1d_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h: KAMA (adaptive moving average) ===
    close = prices['close'].values
    # Efficiency Ratio: |price change over 10 periods| / sum of absolute changes
    change = np.abs(np.diff(close, 10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # sum of absolute changes
    # Handle the array dimensions properly
    er = np.zeros_like(close)
    er[10:] = change[9:] / (np.sum(np.abs(np.diff(close[:10+np.arange(len(change))])), axis=0) + 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily: Volume regime (high/low volume days) ===
    vol_1d = df_1d['volume'].values
    # Volume percentile rank over 50-day lookback
    vol_rank = np.zeros_like(vol_1d)
    for i in range(len(vol_1d)):
        if i < 10:
            vol_rank[i] = 0.5
        else:
            vol_rank[i] = np.sum(vol_1d[max(0, i-49):i+1] <= vol_1d[i]) / min(i+1, 50)
    # High volume regime: vol_rank > 0.7, Low volume regime: vol_rank < 0.3
    vol_regime = np.where(vol_rank > 0.7, 1, np.where(vol_rank < 0.3, -1, 0))  # 1=high, -1=low, 0=neutral
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for KAMA
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama[i]
        price = close[i]
        vol_reg = vol_regime_aligned[i]
        
        if position == 0:
            # Long: Price above KAMA AND high volume regime (trending up with participation)
            if price > kama_val and vol_reg == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND high volume regime (trending down with participation)
            elif price < kama_val and vol_reg == 1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA OR low volume regime (loss of momentum)
            if price < kama_val or vol_reg == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA OR low volume regime (loss of momentum)
            if price > kama_val or vol_reg == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
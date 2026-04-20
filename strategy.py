#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_With_Volume_Spike
# Hypothesis: KAMA (adaptive trend) on 4h with 1d trend filter and volume spike confirmation.
# Goes long when KAMA turns up, price above KAMA, 1d trend up, and volume spike (>2x avg).
# Goes short when KAMA turns down, price below KAMA, 1d trend down, and volume spike.
# Uses volume spike to catch momentum bursts and reduce false signals.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: EMA50 trend filter ===
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4h: KAMA (adaptive trend) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    er = np.where(volatility > 0, direction / volatility, 0)  # Efficiency ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Smoothing constant (fast=2, slow=30)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 4h: Volume spike (>2x 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        kama_val = kama[i]
        close_val = close[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(ema50_1d_val) or np.isnan(vol_spike_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, price above KAMA, 1d trend up, volume spike
            if (kama_val > kama[i-1] and  # KAMA turning up
                close_val > kama_val and  # Price above KAMA
                close_val > ema50_1d_val and  # Price above 1d EMA50 (uptrend)
                vol_spike_val):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, price below KAMA, 1d trend down, volume spike
            elif (kama_val < kama[i-1] and  # KAMA turning down
                  close_val < kama_val and  # Price below KAMA
                  close_val < ema50_1d_val and  # Price below 1d EMA50 (downtrend)
                  vol_spike_val):  # Volume spike
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or price below KAMA
            if kama_val < kama[i-1] or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or price above KAMA
            if kama_val > kama[i-1] or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
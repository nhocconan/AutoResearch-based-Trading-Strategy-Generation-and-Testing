#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h KAMA with 1d Volume Spike Confirmation
# - KAMA adapts to market noise, reducing false signals in choppy markets
# - Volume spike (2x 20-period average) confirms institutional participation
# - Long when price crosses above KAMA with volume spike
# - Short when price crosses below KAMA with volume spike
# - Designed to capture trending moves while avoiding whipsaws in ranging markets
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume spike indicator (volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate KAMA on 6h timeframe
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Avoid division by zero
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start after KAMA warmup
        # Skip if NaN in indicators
        if np.isnan(kama[i]) or np.isnan(volume_spike_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_spike = volume_spike_6h[i]
        
        if position == 0:
            # Long entry: price crosses above KAMA with volume spike
            if price > kama_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below KAMA with volume spike
            elif price < kama_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_KAMA_VolumeSpike"
timeframe = "6h"
leverage = 1.0
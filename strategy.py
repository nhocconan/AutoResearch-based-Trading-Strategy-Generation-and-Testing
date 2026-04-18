#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter_v3
Hypothesis: KAMA adapts to market noise, providing a robust trend filter. Combined with RSI(14) for momentum confirmation and volume spike for institutional participation, this strategy aims to capture high-probability trend continuations in both bull and bear markets. The adaptive nature of KAMA reduces whipsaws during sideways markets, while RSI filters out extreme conditions. Designed for low trade frequency to minimize fee drag.
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
    
    # KAMA (Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    er[:] = np.where(volatility != 0, change / volatility, 0)
    # Pad to match length
    er = np.concatenate([np.full(er_len, np.nan), er])
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # Initialize
    for i in range(er_len + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # Already on 4h, but using for consistency
    rsi_aligned = align_htf_to_ltf(prices, prices, rsi)
    volume_spike_aligned = align_htf_to_ltf(prices, prices, volume_spike)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Ensure indicators are valid
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50 (bullish momentum), volume spike
            if price > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50 (bearish momentum), volume spike
            elif price < kama_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAMA or RSI < 40 (loss of momentum)
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAMA or RSI > 60 (loss of momentum)
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter_v3"
timeframe = "4h"
leverage = 1.0
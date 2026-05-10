#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: Use 1d KAMA to determine trend direction (adaptive moving average), enter long/short when price crosses above/below KAMA with volume confirmation.
Exit when price crosses back or trend reverses. Works in bull/bear by following adaptive trend.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # SC for fastest EMA
    slow_sc = 2 / (30 + 1)  # SC for slowest EMA
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=er_period))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.zeros_like(close_1d)
    er[er_period:] = change[er_period:] / volatility[er_period:]
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[er_period] = close_1d[er_period]
    for i in range(er_period + 1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Volume SMA20 for confirmation
    vol_sma20 = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_sma20[i] = np.mean(volume_1d[i-20:i])
    
    # Align to 1d timeframe (no alignment needed as we are already on 1d)
    kama_aligned = kama
    vol_sma20_aligned = vol_sma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20) + 1  # Wait for KAMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(vol_sma20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_sma20_aligned[i]
        
        # Trend and price relative to KAMA
        is_uptrend = close[i] > kama_aligned[i]
        is_downtrend = close[i] < kama_aligned[i]
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        if position == 0:
            # Long: price crosses above KAMA, in uptrend, with volume
            if price_above_kama and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA, in downtrend, with volume
            elif price_below_kama and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back below KAMA or trend turns down
            if not price_above_kama or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above KAMA or trend turns up
            if not price_below_kama or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_1d_KAMA_Direction_Volume_Trend
# Hypothesis: 12h timeframe with KAMA trend direction (1d) and volume confirmation.
# Uses daily KAMA direction for trend bias, reducing counter-trend trades.
# Volume surge (2x 24-period MA) confirms institutional participation.
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear by requiring trend alignment, avoiding chop whipsaws.

name = "12h_1d_KAMA_Direction_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (volatility.cumsum() - np.concatenate([[0], volatility.cumsum()[:-1]]))
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume average (24-period for 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for KAMA + vol MA
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1d close > KAMA
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > kama_aligned[i]
        downtrend = close_1d_aligned[i] < kama_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price above KAMA in uptrend with volume spike
            if close[i] > kama_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA in downtrend with volume spike
            elif close[i] < kama_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price below KAMA or trend fails
                if close[i] < kama_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price above KAMA or trend fails
                if close[i] > kama_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
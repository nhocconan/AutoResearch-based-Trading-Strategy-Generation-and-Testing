#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume
Hypothesis: Daily Camarilla R1/S1 breakout in direction of weekly EMA21 trend with volume confirmation.
Works in bull/bear by following higher timeframe trend. Target: 10-25 trades/year.
"""

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[:21])
        alpha = 2 / (21 + 1)
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Camarilla levels (based on previous day)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    # Calculate pivot from previous day's OHLC
    for i in range(1, n):
        if not np.isnan(high[i-1]) and not np.isnan(low[i-1]) and not np.isnan(close[i-1]):
            pivot = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            range_ = high[i-1] - low[i-1]
            camarilla_high[i] = pivot + 1.1 * range_ / 2.0  # R1
            camarilla_low[i] = pivot - 1.1 * range_ / 2.0   # S1
    
    # Volume spike: current volume > 2x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R1 and above weekly EMA21
            if high[i] > camarilla_high[i] and close[i] > ema_21_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 and below weekly EMA21
            elif low[i] < camarilla_low[i] and close[i] < ema_21_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA21
            if close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA21
            if close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
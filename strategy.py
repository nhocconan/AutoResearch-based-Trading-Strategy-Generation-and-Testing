#!/usr/bin/env python3
"""
1d_ElderRay_13_EMA_Power_BullBear_WeeklyTrend
Hypothesis: Elder Ray power indicator (13-period EMA) with weekly trend filter and volume confirmation.
Goes long when Bull Power > 0 and Bear Power < 0 with weekly uptrend, short when Bear Power > 0 and Bull Power < 0 with weekly downtrend.
Uses volume confirmation to avoid false signals. Designed to capture trends in both bull and bear markets by following weekly trend.
Target: 10-20 trades/year per symbol with strict entry conditions to minimize fee drift.
"""

name = "1d_ElderRay_13_EMA_Power_BullBear_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Get weekly trend using 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure volume SMA and EMA13 are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, weekly uptrend, volume confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, weekly downtrend, volume confirmation
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bear Power becomes positive (bullish momentum fading)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bull Power becomes negative (bearish momentum fading)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
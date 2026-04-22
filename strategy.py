#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1-day RSI momentum filter and volume confirmation
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, 
# RSI(14) for momentum strength (>50 for long, <50 for short),
# and volume > 1.5x average for confirmation.
# Designed for 12h timeframe to capture medium-term trends with fewer trades.
# Works in both bull and bear markets by following adaptive trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA(10, 2, 30) on 12h data
    # ER = |Change| / Sum(|abs(change)|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Note: using df_1d for alignment but KAMA is 12h - this is incorrect
    # Fix: Calculate KAMA on 12h data directly
    # Recalculate KAMA properly on 12h data
    change_12h = np.abs(np.diff(close, prepend=close[0]))
    direction_12h = np.abs(np.subtract(close, np.roll(close, 10)))
    # Create volatility array properly
    volatility_12h = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_12h[i] = np.sum(change_12h[i-9:i+1])
    er_12h = np.where(volatility_12h != 0, direction_12h / volatility_12h, 0)
    sc_12h = (er_12h * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_12h = np.full_like(close, np.nan)
    if len(close) > 10:
        kama_12h[9] = close[9]
        for i in range(10, len(close)):
            kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])
    
    # Align RSI from 1d to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start from 10 to ensure KAMA is ready
        # Skip if data not ready
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + volume spike
            if close[i] > kama_12h[i] and rsi_1d_aligned[i] > 50 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI < 50 + volume spike
            elif close[i] < kama_12h[i] and rsi_1d_aligned[i] < 50 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses KAMA in opposite direction
            if position == 1:
                # Exit long: Price below KAMA
                if close[i] < kama_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above KAMA
                if close[i] > kama_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_KAMA_1dRSI_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0
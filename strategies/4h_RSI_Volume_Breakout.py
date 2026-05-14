#!/usr/bin/env python3
# 4h_RSI_Volume_Breakout
# Hypothesis: RSI(14) extremes with volume confirmation on 4h timeframe. In bull markets, RSI > 70 with volume indicates strong momentum; in bear markets, RSI < 30 with volume indicates capitulation and potential reversal. Volume filter ensures institutional participation, reducing false signals. Designed for 20-40 trades/year to minimize fee drag. Works in both bull and bear by capturing momentum bursts and reversal attempts.

name = "4h_RSI_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if RSI or volume data is not ready
        if i < 14 or np.isnan(rsi[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 70 (overbought momentum) with volume spike
            if rsi[i] > 70 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 30 (oversold capitulation) with volume spike
            elif rsi[i] < 30 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI falls below 50 (momentum fading)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI rises above 50 (selling pressure fading)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
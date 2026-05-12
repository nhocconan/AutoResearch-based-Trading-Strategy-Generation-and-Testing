#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_Volume_Confirm
Hypothesis: KAMA trend direction from 4h data combined with RSI(14) overbought/oversold conditions and volume confirmation (1.5x average) captures momentum reversals in both bull and bear markets. Works by entering when price deviates from trend but shows exhaustion, avoiding whipsaws in choppy markets.
"""

name = "4h_KAMA_RSI_Trend_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # KAMA calculation
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        direction = np.abs(np.subtract(close, np.roll(close, length)))
        volatility = np.sum(change, axis=0) if change.ndim > 1 else np.sum(change)
        # Handle 1D arrays
        if change.ndim == 1:
            volatility = np.cumsum(change)
            volatility = np.where(np.arange(len(change)) >= length, 
                                volatility - np.roll(volatility, length), 
                                volatility)
        er = np.where(volatility != 0, direction / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_vals = kama(close, length=10, fast=2, slow=30)

    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals

    rsi_vals = rsi(close, length=14)

    # Volume confirmation
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = vol_ma[10]
    vol_ma[-10:] = vol_ma[-11]
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0

    for i in range(20, n):
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below KAMA (pullback in uptrend) + RSI oversold + volume spike
            if (close[i] < kama_vals[i] and 
                rsi_vals[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above KAMA (pullback in downtrend) + RSI overbought + volume spike
            elif (close[i] > kama_vals[i] and 
                  rsi_vals[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above KAMA (trend resumption) or RSI overbought
            if close[i] > kama_vals[i] or rsi_vals[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below KAMA (trend resumption) or RSI oversold
            if close[i] < kama_vals[i] or rsi_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
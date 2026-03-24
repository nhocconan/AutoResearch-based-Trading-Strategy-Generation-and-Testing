#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC Cap Dominance RSI Strategy (Adapted)"
timeframe = "4h"
leverage = 1

def calculate_rsi(series, length):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) < length:
        return np.zeros_like(gain)
    avg_gain[length-1] = np.mean(gain[:length])
    avg_loss[length-1] = np.mean(loss[:length])
    for i in range(length, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
    rs = np.zeros_like(avg_gain)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    if prices.empty:
        return np.array([])
    src = (prices['high'] + prices['low'] + prices['close']) / 3.0
    length = 7
    bull_level = 56.0
    bear_level = 37.0
    rsi = calculate_rsi(src, length)
    signals = np.zeros(len(prices))
    current_pos = 0
    for i in range(1, len(prices)):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
            signals[i] = current_pos
            continue
        long_cond = rsi[i] > bull_level and rsi[i-1] <= bull_level
        short_cond = rsi[i] < bear_level and rsi[i-1] >= bear_level
        if long_cond:
            current_pos = 1
        elif short_cond:
            current_pos = -1
        signals[i] = current_pos
    signals[0] = 0
    return signals

#!/usr/bin/env python3
# 4h_RSI_Divergence_Volume_Squeeze
# Hypothesis: RSI divergence (bullish/bearish) combined with Bollinger Band squeeze and volume confirmation provides high-probability reversals in both bull and bear markets. The squeeze indicates low volatility priming for expansion, while RSI divergence signals exhaustion. Volume confirms institutional interest. Targets 20-30 trades/year to minimize fee drag.

name = "4h_RSI_Divergence_Volume_Squeeze"
timeframe = "4h"
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

    # Get 4h data for Bollinger Bands and RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate RSI (14)
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='full')[:len(arr)]
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='full')[:len(arr)]
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # First period values are invalid
        rsi[:period] = np.nan
        return rsi

    rsi_vals = rsi(close_4h, 14)

    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    bb_width = (upper - lower) / sma  # Normalized width

    # Bollinger Band squeeze: width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma

    # RSI divergence detection
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    lookback = 5  # Look for divergence over last 5 periods

    for i in range(lookback, n):
        if np.isnan(rsi_vals[i]) or np.isnan(rsi_vals[i-lookback]):
            continue
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (low[i] < low[i-lookback] and 
            rsi_vals[i] > rsi_vals[i-lookback]):
            bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (high[i] > high[i-lookback] and 
            rsi_vals[i] < rsi_vals[i-lookback]):
            bearish_div[i] = True

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_avg_20 * 1.5)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_vals[i]) or np.isnan(sma[i]) or 
            np.isnan(std[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish divergence + squeeze + volume confirmation
            if bullish_div[i] and squeeze[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + squeeze + volume confirmation
            elif bearish_div[i] and squeeze[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 or breakdown of squeeze (volatility expansion)
            if rsi_vals[i] > 70 or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 or breakdown of squeeze
            if rsi_vals[i] < 30 or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
# 12h_KAMA_RSI_Chop_Trend
# Hypothesis: Combine KAMA trend direction with RSI momentum and Choppiness index regime filter.
# Long when KAMA trending up, RSI > 50, and market is trending (CHOP < 38.2).
# Short when KAMA trending down, RSI < 50, and market is trending (CHOP < 38.2).
# Uses 1d timeframe for KAMA and RSI calculation, 12h for Choppiness index to avoid look-ahead.
# Designed for 12h timeframe to target 12-37 trades/year, avoiding excessive trading.
# Works in both bull and bear markets by following the trend with momentum confirmation.

name = "12h_KAMA_RSI_Chop_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (adaptive moving average) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period volatility
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    er[10:] = change[9:] / np.where(volatility[9:] == 0, 1, volatility[9:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI on 1d close
    delta = np.diff(close_1d, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50
    
    # Get 12h data for Choppiness index calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Choppiness Index (14-period)
    atr_12h = np.zeros_like(high_12h)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]),
                               np.abs(low_12h[1:] - close_12h[:-1])))
    atr_12h[1:] = tr
    # Smooth ATR
    atr_smoothed = np.zeros_like(atr_12h)
    atr_smoothed[1] = atr_12h[1]
    for i in range(2, len(atr_12h)):
        atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr_12h[i]) / 14
    # Calculate sum of true ranges over 14 periods
    tr_sum = np.zeros_like(high_12h)
    tr_sum[13] = np.sum(tr[:14])
    for i in range(14, len(tr_sum)):
        tr_sum[i] = tr_sum[i-1] - tr[i-13] + tr[i]
    # Choppiness Index
    chop = np.zeros_like(high_12h)
    for i in range(13, len(high_12h)):
        if tr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_smoothed[i] / tr_sum[i]) / np.log10(14)
        else:
            chop[i] = 50
    # Handle first 13 values
    chop[:13] = 50
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up, RSI > 50, trending market (CHOP < 38.2)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, trending market (CHOP < 38.2)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down or RSI < 50 or choppy market (CHOP > 61.8)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up or RSI > 50 or choppy market (CHOP > 61.8)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
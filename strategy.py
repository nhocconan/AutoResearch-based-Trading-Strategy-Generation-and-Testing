#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
# Combined with RSI for momentum confirmation and Choppiness Index to avoid false signals in low-volatility chop.
# Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50).
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "12h_KAMA_Trend_RSI_Chop_Filter"
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

    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14-period)
    atr_1d = []
    tr_1d = np.maximum(np.maximum(
        df_1d['high'][1:] - df_1d['low'][1:],
        np.abs(df_1d['high'][1:] - df_1d['close'][:-1]),
        np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    ), 0)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    high_max_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_min_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10((atr_1d * 14) / (high_max_1d - low_min_1d)) / np.log10(14)
    chop = np.where((high_max_1d - low_min_1d) == 0, 50, chop_raw)
    
    # Calculate KAMA on 12h
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio
    change_12h = np.abs(np.diff(close_12h, n=10))
    change_12h = np.concatenate([[np.nan]*10, change_12h])
    volatility_12h = np.abs(np.diff(close_12h, n=1))
    volatility_12h = np.concatenate([[np.nan], volatility_12h])
    vol_sum_12h = pd.Series(volatility_12h).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum_12h != 0, change_12h / vol_sum_12h, 0)
    
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        if np.isnan(kama_12h[i-1]) or np.isnan(sc[i]):
            kama_12h[i] = close_12h[i]
        else:
            kama_12h[i] = kama_12h[i-1] + sc[i] * (close_12h[i] - kama_12h[i-1])
    
    # Calculate RSI on 12h
    delta_12h = np.diff(close_12h)
    delta_12h = np.concatenate([[np.nan], delta_12h])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs_12h = np.where(avg_loss_12h != 0, avg_gain_12h / avg_loss_12h, 0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # Align indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(kama_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up + RSI > 50 + Chop < 61.8 (trending market)
            if kama_12h_aligned[i] > close[i] and rsi_12h_aligned[i] > 50 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down + RSI < 50 + Chop < 61.8 (trending market)
            elif kama_12h_aligned[i] < close[i] and rsi_12h_aligned[i] < 50 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down or RSI < 50 or Chop > 61.8 (choppy market)
            if kama_12h_aligned[i] < close[i] or rsi_12h_aligned[i] < 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up or RSI > 50 or Chop > 61.8 (choppy market)
            if kama_12h_aligned[i] > close[i] or rsi_12h_aligned[i] > 50 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
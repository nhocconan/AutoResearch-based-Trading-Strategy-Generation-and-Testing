#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction on daily timeframe.
# Enter long when price is above KAMA and RSI(14) > 50; enter short when price is below KAMA and RSI(14) < 50.
# Uses 1-week timeframe for trend confirmation: only take signals when price is above/below weekly EMA(20).
# Designed for low-frequency trading (10-25 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate KAMA(10,2,30) on daily close
    er = np.full_like(close, np.nan)
    if len(close) >= 10:
        for i in range(9, len(close)):
            if close[i-9] != 0:
                er[i] = abs(close[i] - close[i-9]) / np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    sc = np.where(~np.isnan(er), (er * (0.6645 - 0.0645) + 0.0645) ** 2, 0)
    kama = np.full_like(close, np.nan)
    if len(close) >= 10:
        kama[9] = close[9]
        for i in range(10, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20)
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need enough data for KAMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price above KAMA, RSI > 50, and above weekly EMA (bullish alignment)
            if close[i] > kama[i] and rsi[i] > 50 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA, RSI < 50, and below weekly EMA (bearish alignment)
            elif close[i] < kama[i] and rsi[i] < 50 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below KAMA or RSI drops below 50
            if close[i] < kama[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA or RSI rises above 50
            if close[i] > kama[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
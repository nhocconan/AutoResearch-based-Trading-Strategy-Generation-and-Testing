#!/usr/bin/env python3
# 1D_KAMA_RSI_Chop_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
# Long when: KAMA trending up, RSI > 50, and market is trending (CHOP < 38.2).
# Short when: KAMA trending down, RSI < 50, and market is trending (CHOP < 38.2).
# Works in bull/bear by following adaptive trend and avoiding choppy markets.
# Target: 15-25 trades/year per symbol.

name = "1D_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[0] = 0
    volatility = abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr_list = []
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = max(tr1, tr2, tr3)
        atr_list.append(tr)
    atr_list = np.array(atr_list)
    atr_sum = pd.Series(atr_list).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) > 0, chop, 50)
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        rsi_above = rsi[i] > 50
        rsi_below = rsi[i] < 50
        trending = chop[i] < 38.2
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + KAMA up + RSI > 50 + trending market
            if weekly_up and kama_up and rsi_above and trending:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + KAMA down + RSI < 50 + trending market
            elif weekly_down and kama_down and rsi_below and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend weakens or RSI reverses
            if not weekly_up or not kama_up or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakens or RSI reverses
            if not weekly_down or not kama_down or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
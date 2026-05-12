#!/usr/bin/env python3
name = "1d_KAMA_Direction_With_RSI_and_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for KAMA and RSI
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # KAMA components on weekly
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.subtract(close_1w[10:], close_1w[:-10]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan, dtype=float)
    kama[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # RSI(14) on weekly
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Choppiness Index on weekly (14-period)
    atr1 = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(np.subtract(high_1w[1:], np.append([close_1w[0]], close_1w[:-1]))),
                                 np.abs(np.subtract(low_1w[1:], np.append([close_1w[0]], close_1w[:-1])))))
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align all to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + Chop < 61.8 (trending)
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi_1w_aligned[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + Chop < 61.8 (trending)
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_1w_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or Chop > 61.8 (choppy)
            if kama_aligned[i] < kama_aligned[i-1] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or Chop > 61.8 (choppy)
            if kama_aligned[i] > kama_aligned[i-1] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
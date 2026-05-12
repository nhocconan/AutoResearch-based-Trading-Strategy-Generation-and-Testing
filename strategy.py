#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== Weekly Trend Filter (1w) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # ===== Daily KAMA Direction (Primary) =====
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    er[9:] = change[9:] / (volatility[9:] + 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[9] = close[9]  # Start after ER calculation
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ===== Daily RSI(14) =====
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== Daily Choppiness Index (14) =====
    atr1 = np.zeros_like(close)
    tr1 = np.maximum(high[1:] - low[1:], 
                     np.maximum(np.abs(high[1:] - close[:-1]), 
                                np.abs(low[1:] - close[:-1])))
    atr1[1:] = pd.Series(tr1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # True range for period
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    
    chop = np.zeros_like(close)
    chop[13:] = 100 * np.log10(sum_tr14[13:] / sum_atr14[13:]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up (bullish) + RSI > 50 + Chop < 61.8 (trending)
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (bearish) + RSI < 50 + Chop < 61.8 (trending)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down OR RSI < 40 OR Chop > 61.8 (ranging)
            if (kama[i] < kama[i-1] or 
                rsi[i] < 40 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up OR RSI > 60 OR Chop > 61.8 (ranging)
            if (kama[i] > kama[i-1] or 
                rsi[i] > 60 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
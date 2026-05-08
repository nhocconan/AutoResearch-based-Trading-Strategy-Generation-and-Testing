#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + 1d RSI mean reversion + 1d chop regime filter.
# Long when KAMA trend is up (price > KAMA) AND 1d RSI < 30 (oversold) AND 1d chop > 61.8 (ranging market).
# Short when KAMA trend is down (price < KAMA) AND 1d RSI > 70 (overbought) AND 1d chop > 61.8.
# Exit when price crosses back below/above KAMA.
# Uses 12h timeframe with 1d RSI and chop filter for higher timeframe context.
# Designed to work in both bull (mean reversion in ranges) and bear (avoid trends via chop filter).

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for RSI and chop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # KAMA on 12h data
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Correct ER calculation
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if np.sum(np.abs(np.diff(close[i-er_period+1:i+1]))) > 0:
            er[i] = np.abs(close[i] - close[i-er_period]) / np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
        else:
            er[i] = 0
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    close_d = df_d['close'].values
    delta = np.diff(close_d, prepend=close_d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50
    
    # Daily Chop(14)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_h = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_h - min_l)) / np.log10(14)
    chop[np.isnan(chop)] = 50
    chop[max_h == min_l] = 50  # Avoid division by zero
    
    # Align RSI and Chop to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_d, chop)
    
    # Conditions
    kama_up = close > kama
    kama_down = close < kama
    rsi_oversold = rsi_aligned < 30
    rsi_overbought = rsi_aligned > 70
    chop_high = chop_aligned > 61.8  # Ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Sufficient warmup for KAMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, RSI oversold, choppy market
            long_cond = kama_up[i] and rsi_oversold[i] and chop_high[i]
            # Short conditions: KAMA down, RSI overbought, choppy market
            short_cond = kama_down[i] and rsi_overbought[i] and chop_high[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
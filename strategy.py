#!/usr/bin/env python3
# 1d_1w_kama_rsi_v1
# Hypothesis: Use 1-week EMA34 for trend, daily KAMA(14) for momentum, and daily RSI(14) for entries. 
# Long when weekly trend is up, price > KAMA, and RSI < 40 (oversold bounce).
# Short when weekly trend is down, price < KAMA, and RSI > 60 (overbought rejection).
# Exit on opposite KAMA cross or weekly trend reversal. 
# Target: 10-30 trades/year (40-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily KAMA (14) - Kaufman Adaptive Moving Average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation: |close - close[10]| / sum(|diff|) over 10 periods
    lookback = 10
    change_er = np.zeros_like(close_1d)
    volatility_er = np.zeros_like(close_1d)
    
    for i in range(lookback, len(close_1d)):
        change_er[i] = np.abs(close_1d[i] - close_1d[i-lookback])
        volatility_er[i] = np.sum(np.abs(np.diff(close_1d[i-lookback:i+1])))
    
    er = np.zeros_like(close_1d)
    mask = volatility_er > 0
    er[mask] = change_er[mask] / volatility_er[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 1d timeframe (already aligned since we used daily data)
    kama_aligned = kama  # already on 1d frequency
    rsi_aligned = rsi    # already on 1d frequency
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < KAMA or weekly trend turns down
            if close[i] < kama_aligned[i] or close[i] < ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > KAMA or weekly trend turns up
            if close[i] > kama_aligned[i] or close[i] > ema34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly uptrend, price > KAMA, RSI < 40 (oversold)
            if (close[i] > kama_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                rsi_aligned[i] < 40):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly downtrend, price < KAMA, RSI > 60 (overbought)
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  rsi_aligned[i] > 60):
                position = -1
                signals[i] = -0.25
    
    return signals
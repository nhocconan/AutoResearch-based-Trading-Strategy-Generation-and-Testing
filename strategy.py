#!/usr/bin/env python3
# [24981] 4h_1d_1w_rsi_divergence_v1
# Hypothesis: 4-hour RSI divergence with 1-day and 1-week trend filters. Uses bearish/bullish RSI divergence
# (price makes new high/low but RSI does not) to anticipate reversals. Long when bullish divergence occurs
# in oversold conditions (RSI<30) with 1-day/1-week uptrend (price > EMA50). Short when bearish divergence
# occurs in overbought conditions (RSI>70) with 1-day/1-week downtrend (price < EMA50). Exit when RSI
# returns to neutral (40-60 range). Designed to work in both bull and bear markets by catching
# exhaustion moves at extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_rsi_divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1-day and 1-week data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = np.zeros(len(close_1d))
        ema[0] = close_1d[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1d)):
            ema[i] = alpha * close_1d[i] + (1 - alpha) * ema[i-1]
        ema50_1d[49:] = ema[49:]
    
    # Calculate 1-week EMA50
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema = np.zeros(len(close_1w))
        ema[0] = close_1w[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1w)):
            ema[i] = alpha * close_1w[i] + (1 - alpha) * ema[i-1]
        ema50_1w[49:] = ema[49:]
    
    # Align EMA50 to 4-hour timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral range (40-60)
            if 40 <= rsi_now <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral range (40-60)
            if 40 <= rsi_now <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for bullish RSI divergence (price makes lower low but RSI makes higher low)
            bullish_div = False
            if i >= 2:
                # Look for recent swing low in price
                if low[i] < low[i-1] and low[i] < low[i-2]:
                    # Find prior swing low
                    j = i - 1
                    while j >= 2 and low[j] >= low[j-1]:
                        j -= 1
                    if j >= 2 and low[j] < low[j-1] and low[j] < low[j-2]:
                        # Found two swing lows, check if bullish divergence
                        if low[i] < low[j] and rsi[i] > rsi[j]:
                            bullish_div = True
            
            # Check for bearish RSI divergence (price makes higher high but RSI makes lower high)
            bearish_div = False
            if i >= 2:
                # Look for recent swing high in price
                if high[i] > high[i-1] and high[i] > high[i-2]:
                    # Find prior swing high
                    j = i - 1
                    while j >= 2 and high[j] <= high[j-1]:
                        j -= 1
                    if j >= 2 and high[j] > high[j-1] and high[j] > high[j-2]:
                        # Found two swing highs, check if bearish divergence
                        if high[i] > high[j] and rsi[i] < rsi[j]:
                            bearish_div = True
            
            # Enter long: bullish divergence in oversold with uptrend on both 1D and 1W
            if bullish_div and rsi_now < 30 and close[i] > ema50_1d_aligned[i] and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish divergence in overbought with downtrend on both 1D and 1W
            elif bearish_div and rsi_now > 70 and close[i] < ema50_1d_aligned[i] and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals